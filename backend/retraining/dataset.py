import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io.wavfile as wavfile
import scipy.signal as signal

# ==========================================================
#             ADVANCED DATA AUGMENTATION CLASSES
# ==========================================================

class JPEGCompressionAugment:
    def __init__(self, quality_range=(50, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p
    def __call__(self, img_np):
        if np.random.rand() > self.p:
            return img_np
        quality = np.random.randint(self.quality_range[0], self.quality_range[1])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img_np, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        return decimg

class GaussianBlurAugment:
    def __init__(self, kernel_sizes=[3, 5], p=0.5):
        self.kernel_sizes = kernel_sizes
        self.p = p
    def __call__(self, img_np):
        if np.random.rand() > self.p:
            return img_np
        ksize = np.random.choice(self.kernel_sizes)
        return cv2.GaussianBlur(img_np, (ksize, ksize), 0)

class ResizeCompressionAugment:
    def __init__(self, scale_range=(0.5, 0.9), p=0.5):
        self.scale_range = scale_range
        self.p = p
    def __call__(self, img_np):
        if np.random.rand() > self.p:
            return img_np
        h, w = img_np.shape[:2]
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        small = cv2.resize(img_np, (int(w * scale), int(h * scale)))
        return cv2.resize(small, (w, h))

class BrightnessShiftAugment:
    def __init__(self, shift_range=(-30, 30), p=0.5):
        self.shift_range = shift_range
        self.p = p
    def __call__(self, img_np):
        if np.random.rand() > self.p:
            return img_np
        shift = np.random.randint(self.shift_range[0], self.shift_range[1])
        # Simple RGB brightness shift with clipping
        img_shifted = img_np.astype(int) + shift
        return np.clip(img_shifted, 0, 255).astype(np.uint8)

class RandomCropAugment:
    def __init__(self, crop_size=(224, 224), target_size=(256, 256), p=0.5):
        self.crop_size = crop_size
        self.target_size = target_size
        self.p = p
    def __call__(self, img_np):
        if np.random.rand() > self.p:
            return img_np
        h, w = img_np.shape[:2]
        ch, cw = self.crop_size
        if h > ch and w > cw:
            y = np.random.randint(0, h - ch)
            x = np.random.randint(0, w - cw)
            crop = img_np[y:y+ch, x:x+cw]
            return cv2.resize(crop, self.target_size)
        return img_np

class HorizontalFlipAugment:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img_np):
        if np.random.rand() > self.p:
            return img_np
        return cv2.flip(img_np, 1)

# ==========================================================
#                   DATASET IMPLEMENTATIONS
# ==========================================================

class DeepfakeImageDataset(Dataset):
    """
    Advanced PyTorch Dataset supporting simultaneous training from multiple datasets:
    CIFAKE, FaceForensics++, Celeb-DF, DFDC, and DiffusionDB.
    """
    def __init__(self, root_dir, augment=False, is_dummy=False, num_dummy_samples=50):
        self.root_dir = root_dir
        self.augment = augment
        self.samples = []
        self.is_dummy = is_dummy

        # Set up augmentations
        self.aug_pipeline = [
            RandomCropAugment(p=0.4),
            BrightnessShiftAugment(p=0.4),
            ResizeCompressionAugment(p=0.3),
            GaussianBlurAugment(p=0.3),
            JPEGCompressionAugment(p=0.4),
            HorizontalFlipAugment(p=0.5)
        ]

        if is_dummy:
            self._create_dummy_image_dataset(num_dummy_samples)
        else:
            if not os.path.exists(root_dir):
                raise ValueError(f"Image dataset root '{root_dir}' not found. Production training requires real datasets.")
            else:
                self._load_multi_datasets()

    def _load_multi_datasets(self):
        # The expected structure under data/images:
        # data/images/
        # ├── cifake/ (REAL, FAKE)
        # ├── diffusiondb/ (REAL, FAKE)
        # ├── faceforensics/ (real, fake)
        # ├── celeb-df/ (bonafide, spoof)
        # └── dfdc/ (real, fake)
        
        dataset_specs = {
            "cifake": {"real": ["REAL", "real"], "fake": ["FAKE", "fake"]},
            "diffusiondb": {"real": ["REAL", "real"], "fake": ["FAKE", "fake"]},
            "faceforensics": {"real": ["real", "REAL"], "fake": ["fake", "FAKE"]},
            "celeb-df": {"real": ["bonafide"], "fake": ["spoof"]},
            "dfdc": {"real": ["real", "REAL"], "fake": ["fake", "FAKE"]}
        }
        
        loaded_count = {}
        
        if os.path.exists(self.root_dir):
            for name, spec in dataset_specs.items():
                dataset_path = os.path.join(self.root_dir, name)
                if not os.path.exists(dataset_path):
                    # Try finding case-insensitive match
                    for entry in os.listdir(self.root_dir):
                        if entry.lower() == name.lower() and os.path.isdir(os.path.join(self.root_dir, entry)):
                            dataset_path = os.path.join(self.root_dir, entry)
                            break
                            
                if os.path.exists(dataset_path) and os.path.isdir(dataset_path):
                    loaded_count[name] = 0
                    
                    # Scan for Real
                    real_dir = None
                    for cat in spec["real"]:
                        tmp_dir = os.path.join(dataset_path, cat)
                        if not os.path.exists(tmp_dir):
                            for entry in os.listdir(dataset_path):
                                if entry.lower() == cat.lower() and os.path.isdir(os.path.join(dataset_path, entry)):
                                    tmp_dir = os.path.join(dataset_path, entry)
                                    break
                        if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
                            real_dir = tmp_dir
                            break
                            
                    if real_dir:
                        for root, _, files in os.walk(real_dir):
                            for filename in files:
                                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                                    self.samples.append((os.path.join(root, filename), 0.0))
                                    loaded_count[name] += 1
                                    
                    # Scan for Fake
                    fake_dir = None
                    for cat in spec["fake"]:
                        tmp_dir = os.path.join(dataset_path, cat)
                        if not os.path.exists(tmp_dir):
                            for entry in os.listdir(dataset_path):
                                if entry.lower() == cat.lower() and os.path.isdir(os.path.join(dataset_path, entry)):
                                    tmp_dir = os.path.join(dataset_path, entry)
                                    break
                        if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
                            fake_dir = tmp_dir
                            break
                            
                    if fake_dir:
                        for root, _, files in os.walk(fake_dir):
                            for filename in files:
                                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                                    self.samples.append((os.path.join(root, filename), 1.0))
                                    loaded_count[name] += 1

        # Load hard negatives from project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        hard_negatives_root = os.path.join(project_root, "hard_negatives")
        hard_neg_count = 0
        if os.path.exists(hard_negatives_root):
            false_real_dir = os.path.join(hard_negatives_root, "false_real")
            if os.path.exists(false_real_dir):
                for root, _, files in os.walk(false_real_dir):
                    for filename in files:
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                            self.samples.append((os.path.join(root, filename), 1.0))
                            hard_neg_count += 1
            false_fake_dir = os.path.join(hard_negatives_root, "false_fake")
            if os.path.exists(false_fake_dir):
                for root, _, files in os.walk(false_fake_dir):
                    for filename in files:
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                            self.samples.append((os.path.join(root, filename), 0.0))
                            hard_neg_count += 1

        print(f"[+] Loaded dataset statistics:")
        for name, count in loaded_count.items():
            print(f"    - {name}: {count} samples")
        if hard_neg_count > 0:
            print(f"    - Hard Negatives: {hard_neg_count} samples")
            
        print(f"[+] Loaded {len(self.samples)} total samples from active image datasets.")
        
        if not self.samples:
            if self.is_dummy:
                self._create_dummy_image_dataset(50)
            else:
                raise ValueError(
                    f"No images found in {self.root_dir} or hard_negatives/. "
                    "Production training requires a real dataset. "
                    "Please check your dataset paths."
                )

    def _create_dummy_image_dataset(self, num_samples):
        print(f"[+] Generating {num_samples} in-memory simulated image samples...")
        for i in range(num_samples):
            label = float(i % 2)  # Alternate REAL/FAKE
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            if label == 0.0:
                # REAL image: high frequency details
                noise = np.random.normal(127, 40, (256, 256, 3)).astype(np.uint8)
                img = cv2.addWeighted(img, 0.5, noise, 0.5, 0)
                cv2.line(img, (10, 10), (240, 240), (255, 255, 255), 1)
            else:
                # FAKE image: smooth skin texture + boundary blending discrepancy
                smooth = np.random.normal(127, 5, (256, 256, 3)).astype(np.uint8)
                img = cv2.addWeighted(img, 0.9, smooth, 0.1, 0)
                img = cv2.GaussianBlur(img, (7, 7), 0)
                cv2.rectangle(img, (50, 50), (206, 206), (0, 0, 255), 2)
            self.samples.append((img, label))

    def get_sample_weights(self):
        """
        Calculate sample weights for WeightedRandomSampler to handle class imbalance.
        """
        labels = [label for _, label in self.samples]
        class_counts = np.bincount(np.array(labels, dtype=int))
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[int(label)] for label in labels]
        return sample_weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_data, label = self.samples[idx]
        
        if isinstance(img_data, str):
            img = cv2.imread(img_data)
            if img is None:
                img = np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                img = cv2.resize(img, (256, 256))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = img_data

        # Apply custom augmentations if enabled and not dummy data (dummy doesn't need heavy augs)
        if self.augment and not self.is_dummy:
            for aug in self.aug_pipeline:
                img = aug(img)

        # Normalize and convert to PyTorch tensor format (C, H, W)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img_tensor, torch.tensor([label], dtype=torch.float32)


class DeepfakeAudioDataset(Dataset):
    """
    Advanced PyTorch Dataset for Deepfake Audio supporting ASVspoof and WaveFake.
    Computes Mel-spectrograms on the fly and handles class imbalance.
    """
    def __init__(self, root_dir, is_dummy=False, num_dummy_samples=40):
        self.root_dir = root_dir
        self.samples = []
        self.is_dummy = is_dummy

        if is_dummy:
            self._create_dummy_audio_dataset(num_dummy_samples)
        else:
            if not os.path.exists(root_dir):
                print(f"[!] Warning: Audio dataset root '{root_dir}' not found. Switching to dummy generator.")
                self.is_dummy = True
                self._create_dummy_audio_dataset(num_dummy_samples)
            else:
                self._load_multi_datasets()

    def _load_multi_datasets(self):
        supported_datasets = ["asvspoof", "wavefake"]
        subdirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        active_dataset_dirs = [os.path.join(self.root_dir, s) for s in subdirs if s.lower() in supported_datasets]

        if not active_dataset_dirs:
            active_dataset_dirs = [self.root_dir]

        print(f"[*] Scanning audio directories: {active_dataset_dirs}")

        for dataset_path in active_dataset_dirs:
            # Recursively scan for REAL / bonafide / spoof folders
            for label, categories in enumerate([["REAL", "bonafide", "real"], ["FAKE", "spoof", "fake"]]):
                for cat in categories:
                    cat_dir = os.path.join(dataset_path, cat)
                    if not os.path.exists(cat_dir):
                        # Try case insensitive search
                        for d in os.listdir(dataset_path):
                            if d.lower() == cat.lower() and os.path.isdir(os.path.join(dataset_path, d)):
                                cat_dir = os.path.join(dataset_path, d)
                                break
                    
                    if os.path.exists(cat_dir) and os.path.isdir(cat_dir):
                        for root, _, files in os.walk(cat_dir):
                            for filename in files:
                                if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                                    self.samples.append((os.path.join(root, filename), float(label)))

        print(f"[+] Loaded {len(self.samples)} total samples from active audio datasets.")
        if not self.samples:
            print("[!] Warning: No audio found. Switching to dummy generator.")
            self.is_dummy = True
            self._create_dummy_audio_dataset(40)

    def _create_dummy_audio_dataset(self, num_samples):
        print(f"[+] Generating {num_samples} in-memory simulated audio samples...")
        sr = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        for i in range(num_samples):
            label = float(i % 2)
            if label == 0.0:
                # REAL speech: Formant peaks, vocal harmonics
                x = 0.4 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
                envelope = np.sin(np.pi * t)
                x = x * envelope + np.random.normal(0, 0.02, x.shape)
            else:
                # FAKE speech: robotic monotone, uniform spectral energy distribution
                x = 0.6 * np.sin(2 * np.pi * 150 * t) + np.random.normal(0, 0.1, x.shape)
            
            # Normalize
            max_val = np.max(np.abs(x))
            if max_val > 0:
                x = x / max_val
            self.samples.append((x.astype(np.float32), label))

    def get_sample_weights(self):
        labels = [label for _, label in self.samples]
        class_counts = np.bincount(np.array(labels, dtype=int))
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[int(label)] for label in labels]
        return sample_weights

    def _compute_mel_spectrogram(self, x, sr=16000):
        frame_size = 512
        hop_size = 256
        n_mels = 256
        
        f, t, Sxx = signal.spectrogram(x, fs=sr, nperseg=frame_size, noverlap=frame_size - hop_size)
        
        mel_min = 0
        mel_max = 2595 * np.log10(1 + (sr/2) / 700)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        bin_indices = np.floor((frame_size + 1) * hz_points / sr).astype(int)
        
        filters = np.zeros((n_mels, len(f)))
        for i in range(1, n_mels + 1):
            for j in range(bin_indices[i-1], bin_indices[i]):
                filters[i-1, j] = (j - bin_indices[i-1]) / (bin_indices[i] - bin_indices[i-1])
            for j in range(bin_indices[i], bin_indices[i+1]):
                filters[i-1, j] = (bin_indices[i+1] - j) / (bin_indices[i+1] - bin_indices[i])
                
        mel_spec = np.dot(filters, Sxx)
        log_mel_spec = np.log10(mel_spec + 1e-10)
        
        log_mel_spec_resized = cv2.resize(log_mel_spec, (256, 256))
        return log_mel_spec_resized

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_data, label = self.samples[idx]
        sr = 16000
        
        if isinstance(audio_data, str):
            try:
                fs, x = wavfile.read(audio_data)
                if len(x.shape) > 1:
                    x = np.mean(x, axis=1)
                x = x.astype(np.float32)
                if fs != sr:
                    num_samples = int(len(x) * sr / fs)
                    x = signal.resample(x, num_samples)
            except Exception:
                x = np.zeros(sr)
        else:
            x = audio_data

        mel_spec = self._compute_mel_spectrogram(x, sr)
        mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
        return mel_tensor, torch.tensor([label], dtype=torch.float32)
