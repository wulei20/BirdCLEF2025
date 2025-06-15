import torch
import torchaudio
import numpy as np
import colorednoise
from torch_audiomentations import Compose, PitchShift, Shift
from audiomentations import Gain, AddGaussianNoise, AddGaussianSNR

class AudioProcessor:
    """Factory for audio processing and augmentation operations"""
    def __init__(self, config):
        self.config = config
        self._training_transforms = None
        self._validation_transforms = None
        
    @property
    def training_transforms(self):
        """Lazy initialization of training transforms"""
        if self._training_transforms is None:
            self._training_transforms = RandomTransform([
                NoiseInjectionTransform(p=1, max_noise_level=0.04),
                GaussianNoiseTransform(p=1, min_snr=5, max_snr=20),
                PinkNoiseTransform(p=1, min_snr=5, max_snr=20),
                AddGaussianNoiseTransform(min_amplitude=0.0001, max_amplitude=0.03, p=0.5),
                AddGaussianSNRTransform(min_snr_in_db=5, max_snr_in_db=15, p=0.5),
            ])
        return self._training_transforms
    
    @property
    def validation_transforms(self):
        """Lazy initialization of validation transforms"""
        if self._validation_transforms is None:
            self._validation_transforms = Compose([
                Gain(
                    min_gain_db=-3,
                    max_gain_db=3,
                    p=0.1
                )
            ])
        return self._validation_transforms
    
    def create_spectrogram_converter(self) -> torch.nn.Module:
        """Create spectrogram conversion module"""
        return torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max
        )
    
    def create_db_converter(self) -> torch.nn.Module:
        """Create amplitude to DB converter"""
        return torchaudio.transforms.AmplitudeToDB()
    
    def create_time_mask(self) -> torch.nn.Module:
        """Create time masking transform"""
        return torchaudio.transforms.TimeMasking(
            time_mask_param=self.config.time_mask_param,
            p=0.5
        )
    
    def create_freq_mask(self) -> torch.nn.Module:
        """Create frequency masking transform"""
        return torchaudio.transforms.FrequencyMasking(
            freq_mask_param=self.config.freq_mask_param
        )
    
    def process_audio(self, audio: torch.Tensor, is_training: bool = False) -> torch.Tensor:
        """Apply audio transforms based on training mode"""
        if is_training:
            return self.training_transforms(audio, sample_rate=self.config.sample_rate)
        return self.validation_transforms(audio, sample_rate=self.config.sample_rate)
    
    def normalize_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Normalize spectrogram based on config"""
        if self.config.spectrogram_normalization == "standard":
            return (spectrogram + 80) / 80
        elif self.config.spectrogram_normalization == "custom":
            # Custom normalization logic
            min_val = spectrogram.min()
            max_val = spectrogram.max()
            return (spectrogram - min_val) / (max_val - min_val + 1e-7)
        else:
            return spectrogram
    
    def apply_time_frequency_masking(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply time and frequency masking to spectrogram"""
        spectrogram = self.create_time_mask()(spectrogram)
        if torch.rand(1).item() < 0.5:
            spectrogram = self.create_freq_mask()(spectrogram)
        return spectrogram

class RandomTransform:
    def __init__(self, transforms: list, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, y: np.ndarray):
        if np.random.rand() < self.p:
            transform_idx = np.random.choice(len(self.transforms))
            transform = self.transforms[transform_idx]
            y = transform(y)
        return y

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

class NoiseInjectionTransform(AudioTransformBase):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5, sr=32000):
        super().__init__(always_apply, p)
        self.noise_level = (0.0, max_noise_level)
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented


class GaussianNoiseTransform(AudioTransformBase):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20, sr=32000):
        super().__init__(always_apply, p)
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y**2).max()
        a_noise = a_signal / (10 ** (snr / 20))
        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise**2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoiseTransform(AudioTransformBase):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20, sr=32000):
        super().__init__(always_apply, p)
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y**2).max()
        a_noise = a_signal / (10 ** (snr / 20))
        pink_noise = colorednoise.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise**2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


class AddGaussianNoiseTransform(AudioTransformBase):
    supports_multichannel = True

    def __init__(self, always_apply=False, min_amplitude=0.001, max_amplitude=0.015, p=0.5):
        super().__init__(always_apply, p)
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def apply(self, samples: np.ndarray, sample_rate=32000):
        amplitude = np.random.uniform(self.min_amplitude, self.max_amplitude)
        noise = np.random.randn(*samples.shape).astype(np.float32)
        samples = samples + amplitude * noise
        return samples


class AddGaussianSNRTransform(AudioTransformBase):
    supports_multichannel = True

    def __init__(self, always_apply=False, min_snr_in_db: float = 5.0, max_snr_in_db: float = 40.0, p: float = 0.5):
        super().__init__(always_apply, p)
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db

    def apply(self, samples: np.ndarray, sample_rate=32000):
        snr = np.random.uniform(self.min_snr_in_db, self.max_snr_in_db)
        clean_rms = np.sqrt(np.mean(np.square(samples)))
        a = float(snr) / 20
        noise_rms = clean_rms / (10**a)
        noise = np.random.normal(0.0, noise_rms, size=samples.shape).astype(np.float32)
        return samples + noise
