import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Beta
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import sklearn, timm, torchaudio, os
from torch_audiomentations import Compose, PitchShift, Shift
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Beta
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import sklearn, timm, torchaudio, os
from torch_audiomentations import Compose, PitchShift, Shift


class DataPreprocessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.audio_transforms = Compose(
            [
                PitchShift(
                    min_transpose_semitones=-4,
                    max_transpose_semitones=4,
                    sample_rate=config.sample_rate,
                    p=0.4
                ),
                Shift(min_shift=-0.5, max_shift=0.5, p=0.4)
            ]
        )
        self.time_masking = torchaudio.transforms.TimeMasking(
            time_mask_param=60,
            iid_masks=True,
            p=0.5
        )
        self.frequency_masking = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=24,
            iid_masks=True
        )
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            f_min=config.min_frequency,
            f_max=config.max_frequency,
            n_fft=config.n_fft,
            center=True,
            pad_mode="constant",
            norm="slaney",
            onesided=True,
            mel_scale="slaney"
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=80
        )
        self.frequency_modifier = nn.Parameter(
            torch.from_numpy(
                np.array(
                    [np.concatenate((1 - np.arange(config.n_mels // 2) / (config.n_mels // 2), np.zeros(config.n_mels - config.n_mels // 2)))]
                ).T
            ).float(),
            requires_grad=False
        )

        # 根据设备类型移动模型
        if config.device.type == "cuda":
            self.mel_spectrogram = self.mel_spectrogram.cuda()
            self.frequency_modifier = self.frequency_modifier.cuda()
        else:
            self.mel_spectrogram = self.mel_spectrogram.cpu()
            self.frequency_modifier = self.frequency_modifier.cpu()

    def forward(self, audio: torch.Tensor, training: bool = False) -> torch.Tensor:
        if training:
            audio = self.audio_transforms(audio, sample_rate=self.config.sample_rate)

        spectrogram = self.mel_spectrogram(audio)
        spectrogram = self.amplitude_to_db(spectrogram)

        if self.config.normalization == 80:
            spectrogram = (spectrogram + 80) / 80
        elif self.config.normalization == 255:
            spectrogram = spectrogram / 255
        else:
            raise ValueError("Unsupported normalization method")

        if training:
            spectrogram = self.time_masking(spectrogram)
            if torch.rand(1) < 0.5:
                spectrogram = self.frequency_masking(spectrogram)
            if torch.rand(1) < 0.5:
                random_value = torch.randint(self.config.n_mels // 2, self.config.n_mels, (1,)).item()
                noise = self.frequency_modifier[:random_value] * torch.rand(1).item()
                noise_padded = torch.cat([noise, torch.zeros(self.config.n_mels - random_value).to(noise.device)])
                spectrogram = spectrogram * noise_padded

        return spectrogram


class AttentionMechanism(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str = "linear"):
        super().__init__()
        self.activation = activation
        self.attention = nn.Conv1d(input_dim, output_dim, kernel_size=1)
        self.classifier = nn.Conv1d(input_dim, output_dim, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> tuple:
        attention_weights = torch.softmax(torch.tanh(self.attention(inputs)), dim=-1)
        classifier_output = self._apply_activation(self.classifier(inputs))
        weighted_sum = torch.sum(attention_weights * classifier_output, dim=2)
        return weighted_sum, attention_weights, classifier_output

    def _apply_activation(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.activation == "linear":
            return inputs
        elif self.activation == "sigmoid":
            return torch.sigmoid(inputs)
        else:
            raise ValueError("Unsupported activation function")


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_norm = nn.BatchNorm2d(config.n_mels)
        base_model = timm.create_model(
            config.model_name,
            pretrained=True,
            in_chans=config.input_channels,
            drop_path_rate=0.2,
            drop_rate=0.5
        )
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])
        if "efficientnet" in config.model_name:
            self.input_dim = base_model.classifier.in_features
        elif "eca" in config.model_name:
            self.input_dim = base_model.head.fc.in_features
        elif "res" in config.model_name:
            self.input_dim = base_model.fc.in_features
        else:
            raise ValueError("Unsupported model name")
        self.fc = nn.Linear(self.input_dim, self.input_dim)

    def forward(self, inputs: torch.Tensor) -> tuple:
        inputs = inputs.permute(0, 1, 3, 2)
        frame_count = inputs.shape[2]
        inputs = inputs.transpose(1, 3)
        inputs = self.batch_norm(inputs)
        inputs = inputs.transpose(1, 3)
        inputs = inputs.transpose(2, 3)
        encoded_features = self.encoder(inputs)
        encoded_features = torch.mean(encoded_features, dim=2)
        smoothed_feature1 = F.max_pool1d(encoded_features, kernel_size=3, stride=1, padding=1)
        smoothed_feature2 = F.avg_pool1d(encoded_features, kernel_size=3, stride=1, padding=1)
        smoothed_features = smoothed_feature1 + smoothed_feature2
        smoothed_features = F.dropout(smoothed_features, p=0.5, training=self.training)
        smoothed_features = smoothed_features.transpose(1, 2)
        smoothed_features = F.relu(self.fc(smoothed_features))
        smoothed_features = smoothed_features.transpose(1, 2)
        smoothed_features = F.dropout(smoothed_features, p=0.5, training=self.training)
        return smoothed_features, frame_count


class ModelManager(pl.LightningModule):
    def __init__(self, config, stage: str):
        super().__init__()
        self.config = config
        self.stage = stage
        self.num_classes = len(config.bird_columns)
        self.bird_species = config.bird_columns
        self.data_preprocessor = DataPreprocessor(config)
        self.feature_extractor = FeatureExtractor(config)
        self.attention_mechanism = AttentionMechanism(
            input_dim=self.feature_extractor.input_dim,
            output_dim=self.num_classes,
            activation="sigmoid"
        )
        self.mixup = DataAugmenter(
            beta=config.mixup_beta,
            probability=config.mixup_probability,
            double_mixup_probability=config.double_mixup_probability
        )
        self.mixup2 = DataAugmenter(
            beta=config.mixup_beta2,
            probability=config.mixup2_probability
        )
        self.exponential_moving_average = None
        self.training_step_outputs = []

        # 初始化损失函数
        if config.loss[stage] == "cross_entropy":
            self.loss_function = nn.CrossEntropyLoss(
                label_smoothing=config.label_smoothing,
                reduction="none"
            )
        elif config.loss[stage] == "binary_cross_entropy":
            self.loss_function = nn.BCEWithLogitsLoss(reduction="none")
        else:
            raise ValueError("Unsupported loss function")

    def set_exponential_moving_average(self, ema):
        self.exponential_moving_average = ema

    def on_before_zero_grad(self, *args, **kwargs):
        if self.exponential_moving_average is not None:
            if (self.global_step + 1) % 10 == 0:
                self.exponential_moving_average.update(self)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.learning_rate[self.stage],
            weight_decay=self.config.weight_decay
        )
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.epochs[self.stage],
            T_mult=1,
            eta_min=1e-6,
            last_epoch=-1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1
            }
        }

    def forward(self, batch: tuple) -> tuple:
        inputs, labels, weights = batch
        batch_size, channels, parts = inputs.shape[0], inputs.shape[1], inputs.shape[2]
        if not self.training:
            inputs = inputs.reshape((batch_size * parts, channels, -1))

        if self.training and self.config.mixup:
            inputs, labels, weights = self.mixup(inputs, labels, weights)

        inputs = self.data_preprocessor(inputs, training=self.training)

        if self.training and self.config.mixup2:
            inputs, labels, weights = self.mixup2(inputs, labels, weights)

        features, frame_count = self.feature_extractor(inputs)

        clipwise_output, attention_weights, segmentwise_output = self.attention_mechanism(features)
        logits = torch.sum(attention_weights * self.attention_mechanism.classifier(features), dim=2)
        segmentwise_logits = self.attention_mechanism.classifier(features).transpose(1, 2)
        segmentwise_output = segmentwise_output.transpose(1, 2)

        interpolation_ratio = frame_count // segmentwise_output.size(1)

        framewise_output = self._interpolate(segmentwise_output, interpolation_ratio)
        framewise_output = self._pad_sequence(framewise_output, frame_count)

        framewise_logits = self._interpolate(segmentwise_logits, interpolation_ratio)
        framewise_logits = self._pad_sequence(framewise_logits, frame_count)

        if not self.training:
            clipwise_output = clipwise_output.reshape((batch_size, parts, -1)).max(dim=1).values
            segment_count = segmentwise_logits.shape[1]
            frame_count = framewise_logits.shape[1]
            segmentwise_logits = segmentwise_logits.reshape((batch_size, parts, segment_count, -1)).max(dim=1).values
            framewise_logits = framewise_logits.reshape((batch_size, parts, frame_count, -1)).max(dim=1).values

        loss = 0.5 * self.loss_function(torch.logit(clipwise_output), labels) + 0.5 * self.loss_function(segmentwise_logits.max(1)[0], labels)
        if self.config.loss[self.stage] == "cross_entropy":
            loss = (loss * weights) / weights.sum()
        elif self.config.loss[self.stage] == "binary_cross_entropy":
            loss = loss.sum(dim=1) * weights
        else:
            raise ValueError("Unsupported loss function")

        loss = loss.sum()

        return torch.logit(clipwise_output), labels, loss

    def _interpolate(self, data: torch.Tensor, ratio: int) -> torch.Tensor:
        batch_size, time_steps, features = data.shape
        interpolated_data = data[:, :, None, :].repeat(1, 1, ratio, 1)
        interpolated_data = interpolated_data.reshape(batch_size, time_steps * ratio, features)
        return interpolated_data

    def _pad_sequence(self, sequence: torch.Tensor, target_length: int) -> torch.Tensor:
        return F.interpolate(
            sequence.unsqueeze(1),
            size=(target_length, sequence.size(2)),
            align_corners=True,
            mode="bilinear"
        ).squeeze(1)

    def training_step(self, batch, batch_idx):
        self._freeze_parameters()
        predictions, targets, loss_value = self(batch)
        self.training_step_outputs.append(loss_value)
        return {"loss": loss_value}

    def validation_step(self, batch, batch_idx):
        if self.exponential_moving_average is not None:
            predictions, targets, validation_loss = self.exponential_moving_average.module(batch)
        else:
            predictions, targets, validation_loss = self(batch)
        return {"val_loss": validation_loss, "logits": predictions, "targets": targets}

    def on_validation_epoch_end(self, outputs):
        if len(outputs):
            average_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
            validation_logits = torch.cat([x["logits"] for x in outputs], dim=0).sigmoid().cpu().detach().numpy()
            validation_targets = torch.cat([x["targets"] for x in outputs], dim=0).cpu().detach().numpy()

            validation_df = pd.DataFrame(validation_targets, columns=self.bird_species)
            prediction_df = pd.DataFrame(validation_logits, columns=self.bird_species)

            if self.current_epoch > -1:
                cmap_score_5 = self._calculate_padded_cmap(validation_df, prediction_df, padding_factor=5)
                cmap_score_3 = self._calculate_padded_cmap(validation_df, prediction_df, padding_factor=3)
                average_precision_score = sklearn.metrics.label_ranking_average_precision_score(validation_targets, validation_logits)

                self.log("val_loss", average_loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
                self.log("validation C-MAP score pad 5", cmap_score_5, on_step=False, on_epoch=True, logger=True, prog_bar=True)
                self.log("validation C-MAP score pad 3", cmap_score_3, on_step=False, on_epoch=True, logger=True, prog_bar=True)
                self.log("validation AP score", average_precision_score, on_step=False, on_epoch=True, logger=True, prog_bar=True)

                print(f"epoch {self.current_epoch} validation loss {average_loss}")
                print(f"epoch {self.current_epoch} validation C-MAP score pad 5 {cmap_score_5}")
                print(f"epoch {self.current_epoch} validation C-MAP score pad 3 {cmap_score_3}")
                print(f"epoch {self.current_epoch} validation AP score {average_precision_score}")
            else:
                self.log("val_loss", average_loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
                print(f"epoch {self.current_epoch} validation loss {average_loss}")

            validation_df.to_pickle("val_df.pkl")
            prediction_df.to_pickle("pred_df.pkl")
        else:
            average_loss = 0

        return {"val_loss": average_loss}

    def on_train_epoch_end(self):
        if self.training_step_outputs:
            average_loss = torch.stack(self.training_step_outputs).mean()
            self.training_step_outputs.clear()
            self.log("train_loss", average_loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.exponential_moving_average is not None and (self.current_epoch > self.config.epochs[self.stage] - 3 - 1):
            output_path = self.config.output_path.get(self.stage, ".")
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            torch.save(
                {
                    "state_dict": self.exponential_moving_average.module.state_dict()
                },
                os.path.join(output_path, f"ema_{self.current_epoch}.ckpt")
            )

    def _freeze_parameters(self):
        if self.stage == "finetune":
            self.feature_extractor.eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    @staticmethod
    def _calculate_padded_cmap(solution_df: pd.DataFrame, submission_df: pd.DataFrame, padding_factor: int = 5) -> float:
        padded_solution = solution_df.copy()
        padded_submission = submission_df.copy()
        padding_data = pd.DataFrame([[1] * len(solution_df.columns)] * padding_factor)
        padding_data.columns = solution_df.columns
        padded_solution = pd.concat([padded_solution, padding_data]).reset_index(drop=True)
        padded_submission = pd.concat([padded_submission, padding_data]).reset_index(drop=True)
        return sklearn.metrics.average_precision_score(
            padded_solution.values,
            padded_submission.values,
            average="macro",
        )


class DataAugmenter(nn.Module):
    def __init__(self, beta: float, probability: float, double_mixup_probability: float = 0.0):
        super().__init__()
        self.beta_distribution = Beta(beta, beta)
        self.mixup_probability = probability
        self.double_mixup_probability = double_mixup_probability

    def forward(self, data: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor = None) -> tuple:
        if not self.training:
            return data, labels, weights if weights is not None else (data, labels)

        if torch.rand(1) < self.mixup_probability:
            batch_size = data.shape[0]
            permutation = torch.randperm(batch_size)
            if torch.rand(1) < self.double_mixup_probability:
                data = data + data[permutation]
                labels = labels + labels[permutation]
                labels = torch.clamp(labels, 0, 1)
                if weights is None:
                    return data, labels
                weights = 0.5 * weights + 0.5 * weights[permutation]
                return data, labels, weights
            else:
                second_permutation = torch.randperm(batch_size)
                data = data + data[permutation] + data[second_permutation]
                labels = labels + labels[permutation] + labels[second_permutation]
                labels = torch.clamp(labels, 0, 1)
                if weights is None:
                    return data, labels
                weights = (1 / 3) * weights + (1 / 3) * weights[permutation] + (1 / 3) * weights[second_permutation]
                return data, labels, weights
        else:
            if weights is None:
                return data, labels
            return data, labels, weights


class ModelLoader:
    @staticmethod
    def load_model(config, stage: str, training: bool = True):
        if training:
            model_path = config.model_checkpoint.get(stage)
        else:
            model_path = config.final_model_path

        if model_path is not None:
            state_dict = torch.load(model_path, map_location=config.device)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            print("Loading model from checkpoint")
        else:
            state_dict = None

        model = ModelManager(config, stage)
        if state_dict is not None:
            filtered_state_dict = {k: v for k, v in state_dict.items() if "exponential_moving_average" not in k}
            model.load_state_dict(filtered_state_dict, strict=False)

        if not training:
            model.eval()

        return model
    