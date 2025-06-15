import torch
import os
import librosa
import pandas as pd
import numpy as np
from ast import literal_eval
from audiomentations import Compose, Gain
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class AudioDataPreprocessor:
    def __init__(self, config):
        self.config = config

    def create_audio_transformations(self):
        audio_mentations_transforms = Compose([
            Gain(min_gain_db=-12, max_gain_db=12, p=0.2),
        ])

        custom_audio_transforms = CustomAudioTransformPipeline([
            CustomProbabilityTransform(
                [
                    NoiseInjectionTransform(p=1, max_noise_level=0.04),
                    GaussianNoiseTransform(p=1, min_snr=5, max_snr=20),
                    PinkNoiseTransform(p=1, min_snr=5, max_snr=20),
                    AddGaussianNoiseTransform(min_amplitude=0.0001, max_amplitude=0.03, p=0.5),
                    AddGaussianSNRTransform(min_snr_in_db=5, max_snr_in_db=15, p=0.5),
                ],
                p=0.3,
            ),
        ])
        
        return audio_mentations_transforms, custom_audio_transforms

    def split_dataset(self, dataframe, labels_dataframe):
        train_dataframe = dataframe.copy()
        train_labels_dataframe = labels_dataframe.copy()
        val_dataframe = pd.DataFrame(columns=train_dataframe.columns)
        val_labels_dataframe = pd.DataFrame(columns=train_labels_dataframe.columns)
        return train_dataframe, val_dataframe, train_labels_dataframe, val_labels_dataframe

    def prepare_training_data(self, dataframe):
        # 加载和处理训练数据
        dataframe['secondary_labels'] = dataframe['secondary_labels'].apply(lambda x: literal_eval(x))
        dataframe['path'] = dataframe['filename'].apply(lambda x: os.path.join(self.config.train_path, x))

        # 计算音频时长
        dataframe['duration'] = dataframe['path'].apply(lambda x: librosa.get_duration(path=x))

        # 初始化标签矩阵
        labels_matrix = np.zeros(shape=(len(dataframe), len(self.config.bird_columns)))
        labels_dataframe = pd.DataFrame(labels_matrix, columns=self.config.bird_columns)
        class_sample_count = {col: 0 for col in self.config.bird_columns}
        include_flags = []
        presence_types = []

        # 填充标签并计算类别样本数
        for idx, (primary_label, secondary_labels) in enumerate(zip(dataframe[self.config.primary_label_column].values, dataframe[self.config.secondary_labels_column].values)):
            include = False
            presence = 'background' if primary_label != 'soundscape' else 'soundscape'

            if primary_label in self.config.bird_columns:
                include = True
                presence = 'foreground'
                labels_dataframe.loc[idx, primary_label] = 1
                class_sample_count[primary_label] += 1

            for secondary_label in secondary_labels:
                if secondary_label in self.config.bird_columns:
                    include = True
                    labels_dataframe.loc[idx, secondary_label] = self.config.secondary_label_value
                    class_sample_count[secondary_label] += self.config.secondary_label_weight

            presence_types.append(presence)
            include_flags.append(include)

        # 划分数据集
        train_dataframe, val_dataframe, train_labels_dataframe, val_labels_dataframe = self.split_dataset(dataframe, labels_dataframe)

        # 计算样本权重
        sample_weights = np.zeros(shape=(len(train_dataframe),))
        for idx, (primary_label, secondary_labels) in enumerate(zip(train_dataframe[self.config.primary_label_column].values, train_dataframe[self.config.secondary_labels_column].values)):
            if primary_label in self.config.bird_columns:
                sample_weights[idx] = 1.0 / class_sample_count[primary_label]
            else:
                valid_secondary_labels = [label for label in secondary_labels if label in self.config.bird_columns]
                sample_weights[idx] = np.mean([1.0 / class_sample_count[secondary_label] for secondary_label in valid_secondary_labels])

        return train_dataframe, val_dataframe, train_labels_dataframe, val_labels_dataframe, sample_weights



class BirdAudioDataset(Dataset):
    def __init__(self, dataframe, labels_dataframe, config, resample_type="kaiser_fast", resample=True, is_training=True, pseudo_data=None, transforms=None):
        self.config = config
        self.dataframe = dataframe
        self.labels_dataframe = labels_dataframe
        self.sample_rate = config.sample_rate
        self.n_mels = config.n_mels
        self.f_min = config.f_min
        self.f_max = config.f_max
        self.is_training = is_training
        self.duration = config.duration
        self.audio_length = self.duration * self.sample_rate
        self.resample_type = resample_type
        self.resample = resample
        self.dataframe["weight"] = 1.0
        self.pseudo_data = pseudo_data
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def _adjust_target_labels(self, labels, filename, sample_endpoints, target, version, pseudo_data, pseudo_weights):
        """调整目标标签"""
        adjusted_labels = {label: 0 for label in labels if label in self.config.bird_columns}
        label_components = list(adjusted_labels.keys())

        for oof_data, weight in zip(pseudo_data, pseudo_weights):
            for label in label_components:
                predictions = [oof_data['predictions'][version][filename][label][endpoint] for endpoint in sample_endpoints]
                thresholds = oof_data['thresholds'][label]
                adjustments = np.zeros(shape=(len(predictions),))

                for i, prediction in enumerate(predictions):
                    q3, q2, q1 = thresholds['q3'], thresholds['q2'], thresholds['q1']
                    if prediction >= q3:
                        adjustment = 1.0
                    elif prediction >= q2:
                        adjustment = 0.9
                    elif prediction >= q1:
                        adjustment = 0.5
                    else:
                        adjustment = 0.2
                    adjustments[i] = adjustment

                adjusted_labels[label] += weight * (1 - np.prod(1 - adjustments))

        for label in label_components:
            if adjusted_labels[label] <= 0.6:
                adjusted_labels[label] = 0.01
            elif adjusted_labels[label] <= 0.75:
                adjusted_labels[label] = 0.6
            target[label] = target[label] * adjusted_labels[label]

        return target

    def _load_and_process_audio(self, file_path, target, row):
        """加载并处理音频数据"""
        filename = row['filename']
        labels = [bird for bird in list(set([row[self.config.primary_label_column]] + row[self.config.secondary_labels_column])) if bird in self.config.bird_columns]
        secondary_labels = [bird for bird in row[self.config.secondary_labels_column] if bird in self.config.bird_columns]
        duration = row['duration']

        mixup_part = 1
        work_duration = self.duration * mixup_part
        work_audio_length = work_duration * self.sample_rate

        max_offset = np.max([0, duration - work_duration])
        parts = int(duration // self.config.inference_duration) if duration % self.config.inference_duration == 0 else int(duration // self.config.inference_duration + 1)
        endpoints = [(p + 1) * self.config.inference_duration for p in range(parts)]

        if self.is_training:
            offset = torch.rand((1,)).numpy()[0] * max_offset
            audio_sample, original_sr = librosa.load(file_path, sr=None, mono=True, offset=offset, duration=work_duration)
            
            if (self.resample) and (original_sr != self.sample_rate):
                audio_sample = librosa.resample(audio_sample, original_sr, self.sample_rate, res_type=self.resample_type)

            if len(audio_sample) < work_audio_length:
                audio_sample = self._resample_audio(audio_sample, target_length=work_audio_length)

            audio_sample = audio_sample.reshape((mixup_part, -1))
            audio_sample = np.sum(audio_sample, axis=0)

            if self.transforms is not None:
                audio_sample = self.transforms(audio_sample)

            if len(audio_sample) != self.audio_length:
                audio_sample = self._resample_audio(audio_sample, target_length=self.audio_length)

        else:
            audio, original_sr = librosa.load(file_path, sr=None, mono=True, offset=0, duration=self.config.validation_duration)
            
            if self.resample and original_sr != self.sample_rate:
                audio = librosa.resample(audio, original_sr, self.sample_rate, res_type=self.resample_type)

            audio_chunks = int(np.ceil(len(audio) / self.audio_length))
            audio_sample = [audio[i * self.audio_length:(i + 1) * self.audio_length] for i in range(audio_chunks)]

            if len(audio_sample[-1]) < self.audio_length:
                audio_sample[-1] = self._resample_audio(audio_sample[-1], target_length=self.audio_length)

            val_chunk_count = int(self.config.validation_duration / self.duration)
            if len(audio_sample) > val_chunk_count:
                audio_sample = audio_sample[0:val_chunk_count]
            elif len(audio_sample) < val_chunk_count:
                padding_needed = val_chunk_count - len(audio_sample)
                padding = [np.zeros(shape=(self.audio_length,))] * padding_needed
                audio_sample += padding

            audio_sample = np.stack(audio_sample)

        audio_sample = torch.tensor(audio_sample[np.newaxis]).float()

        target_values = target.values
        if not self.is_training:
            target_values[target_values > 0] = 1

        return audio_sample, target_values

    def _resample_audio(self, audio, target_length, start=None):
        """重采样音频"""
        if len(audio) < target_length:
            audio = np.concatenate([audio, np.zeros(target_length - len(audio))])
            audio = np.concatenate([audio] * int(target_length / len(audio)) + [audio[:(target_length % len(audio))]])
        elif len(audio) > target_length:
            if start is not None:
                audio = audio[start:start + target_length]
            else:
                audio = audio[:target_length]
        return audio

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        target = self.labels_dataframe.loc[index]

        weight = self.dataframe.loc[index, "weight"]
        audio, target = self._load_and_process_audio(self.dataframe.loc[index, "path"], target, row)
        target = torch.tensor(target).float()
        return audio, target, weight


class DataLoaderFactory:
    @staticmethod
    def create_data_loaders(train_dataframe, val_dataframe, train_labels_dataframe, val_labels_dataframe, sample_weights, config, pseudo_data=None, transforms=None):
        """创建数据加载器"""
        num_workers = 0
        sample_weights = torch.from_numpy(sample_weights)
        sampler = WeightedRandomSampler(sample_weights.type('torch.DoubleTensor'), len(sample_weights), replacement=True)

        training_dataset = BirdAudioDataset(
            train_dataframe,
            train_labels_dataframe,
            config,
            is_training=True,
            pseudo_data=pseudo_data,
            transforms=transforms,
        )
        validation_dataset = BirdAudioDataset(
            val_dataframe,
            val_labels_dataframe,
            config,
            is_training=False,
            pseudo_data=None,
            transforms=None,
        )
        training_loader = DataLoader(training_dataset, batch_size=config.batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
        validation_loader = DataLoader(validation_dataset, batch_size=config.test_batch_size, num_workers=num_workers, pin_memory=True)
        return training_loader, validation_loader


class AudioTransformBase:
    def __init__(self, always_apply=False, probability=0.5):
        self.always_apply = always_apply
        self.probability = probability

    def __call__(self, audio: np.ndarray):
        if self.always_apply:
            return self.apply(audio)
        else:
            if np.random.rand() < self.probability:
                return self.apply(audio)
            else:
                return audio

    def apply(self, audio: np.ndarray):
        raise NotImplementedError


class CustomAudioTransformPipeline:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, audio: np.ndarray):
        for transform in self.transforms:
            audio = transform(audio)
        return audio


class CustomProbabilityTransform:
    def __init__(self, transforms: list, probability=1.0):
        self.transforms = transforms
        self.probability = probability

    def __call__(self, audio: np.ndarray):
        if np.random.rand() < self.probability:
            transform_idx = np.random.choice(len(self.transforms))
            transform = self.transforms[transform_idx]
            audio = transform(audio)
        return audio


class NoiseInjectionTransform(AudioTransformBase):
    def __init__(self, always_apply=False, probability=0.5, max_noise_level=0.5, sample_rate=32000):
        super().__init__(always_apply, probability)
        self.noise_level_range = (0.0, max_noise_level)
        self.sample_rate = sample_rate

    def apply(self, audio: np.ndarray, **kwargs):
        noise_level = np.random.uniform(*self.noise_level_range)
        noise = np.random.randn(len(audio))
        augmented_audio = (audio + noise * noise_level).astype(audio.dtype)
        return augmented_audio


class GaussianNoiseTransform(AudioTransformBase):
    def __init__(self, always_apply=False, probability=0.5, min_snr=5, max_snr=20, sample_rate=32000):
        super().__init__(always_apply, probability)
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sample_rate = sample_rate

    def apply(self, audio: np.ndarray, **kwargs):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        signal_amplitude = np.sqrt(audio**2).max()
        noise_amplitude = signal_amplitude / (10 ** (snr / 20))
        white_noise = np.random.randn(len(audio))
        noise_amplitude_white = np.sqrt(white_noise**2).max()
        augmented_audio = (audio + white_noise * 1 / noise_amplitude_white * noise_amplitude).astype(audio.dtype)
        return augmented_audio


class PinkNoiseTransform(AudioTransformBase):
    def __init__(self, always_apply=False, probability=0.5, min_snr=5, max_snr=20, sample_rate=32000):
        super().__init__(always_apply, probability)
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sample_rate = sample_rate

    def apply(self, audio: np.ndarray, **kwargs):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        signal_amplitude = np.sqrt(audio**2).max()
        noise_amplitude = signal_amplitude / (10 ** (snr / 20))
        pink_noise = cn.powerlaw_psd_gaussian(1, len(audio))
        noise_amplitude_pink = np.sqrt(pink_noise**2).max()
        augmented_audio = (audio + pink_noise * 1 / noise_amplitude_pink * noise_amplitude).astype(audio.dtype)
        return augmented_audio


class AddGaussianNoiseTransform(AudioTransformBase):
    supports_multichannel = True

    def __init__(self, always_apply=False, min_amplitude=0.001, max_amplitude=0.015, probability=0.5):
        super().__init__(always_apply, probability)
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def apply(self, audio: np.ndarray, sample_rate=32000):
        amplitude = np.random.uniform(self.min_amplitude, self.max_amplitude)
        noise = np.random.randn(*audio.shape).astype(np.float32)
        audio = audio + amplitude * noise
        return audio


class AddGaussianSNRTransform(AudioTransformBase):
    supports_multichannel = True

    def __init__(self, always_apply=False, min_snr_in_db: float = 5.0, max_snr_in_db: float = 40.0, probability: float = 0.5):
        super().__init__(always_apply, probability)
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db

    def apply(self, audio: np.ndarray, sample_rate=32000):
        snr = np.random.uniform(self.min_snr_in_db, self.max_snr_in_db)
        clean_rms = np.sqrt(np.mean(np.square(audio)))
        a = float(snr) / 20
        noise_rms = clean_rms / (10**a)
        noise = np.random.normal(0.0, noise_rms, size=audio.shape).astype(np.float32)
        return audio + noise
