import os
import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset


class MSCDataset(Dataset):
    """
    Custom Dataset class for Mini Speech Commands.
    
    Args:
        root (str): Root folder path where the dataset is stored.
        classes (list of str): Ordered list of classes specifying the target keywords.
                               This list will be used to map textual labels to integer indices.
        split (str): Dataset split to use ('training', 'validation', or 'testing').
        preprocess (callable, optional): Optional transform to be applied on audio data.
        download (bool): If True, downloads the dataset (placeholder - assumes data exists).
    """
    
    def __init__(self, root, classes, split='training', preprocess=None, download=False):
        self.root = root
        self.classes = classes
        self.split = split
        self.preprocess = preprocess
        self.label_to_idx = {label: idx for idx, label in enumerate(classes)}
        
        # Handle download (placeholder - in real scenario, this would download the dataset)
        if download:
            self._download()
        
        # Map split names to actual folder names (for local environment)
        split_mapping = {
            'training': 'msc-training',
            'validation': 'msc-validation',
            'testing': 'msc-testing'
        }
        
        # Determine the data folder based on split
        if split in split_mapping:
            data_folder = os.path.join(root, split_mapping[split])
            
            if not os.path.exists(data_folder):
                raise FileNotFoundError(
                    f"Dataset folder not found: {data_folder}\n"
                    f"Expected folder structure: {root}/{split_mapping[split]}"
                )
        else:
            data_folder = root
        
        print(f"Using data folder: {data_folder}")
        
        # Collect all audio file paths and their corresponding labels
        self.samples = []
        
        # Check if split folder exists, otherwise use flat structure
        if os.path.exists(data_folder) and os.path.isdir(data_folder):
            # Hierarchical structure (root/msc-train/<label>_*.wav)
            for filename in os.listdir(data_folder):
                if filename.endswith('.wav'):
                    label = filename.split('_')[0]
                    if label in classes:
                        file_path = os.path.join(data_folder, filename)
                        self.samples.append((file_path, label))
        else:
            # flat structure (root/<label>_*.wav)
            for filename in os.listdir(root):
                if filename.endswith('.wav'):
                    label = filename.split('_')[0]
                    if label in classes:
                        file_path = os.path.join(root, filename)
                        self.samples.append((file_path, label))
        
        print(f"Loaded {len(self.samples)} samples from {data_folder} for classes {classes}")
    
    def _download(self):
        """
        Placeholder for dataset download functionality.
        In a real scenario, this would download and extract the dataset.
        """
        pass
        
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Returns a Dictionary with the following structure:
            {
                "x": torch.Tensor - Audio data as a tensor (preprocessed if preprocess is provided),
                "y": int - Integer label corresponding to the keyword
            }
        """
        file_path, label_str = self.samples[idx]
                
        # Load audio using soundfile backend (macOS compatible)
        waveform, sampling_rate = sf.read(file_path, dtype='float32')
        # Convert to torch tensor and ensure shape is (1, samples)
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        
        # Ensure audio is exactly 1 second (16000 samples)
        target_length = 16000
        if waveform.shape[1] < target_length:
            # Pad if too short
            pad_length = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        elif waveform.shape[1] > target_length:
            # Crop if too long
            waveform = waveform[:, :target_length]
        
        # Apply preprocessing if provided
        if self.preprocess is not None:
            waveform = self.preprocess(waveform)
        
        # Convert label string to integer using the provided label mapping
        label_int = self.label_to_idx[label_str]
        
        return {
            "x": waveform,
            "y": label_int
        }
    
    def label_to_int(self, label_str):
        """
        Convert a label string to its corresponding integer index.
        
        Returns the integer index corresponding to the label
        -1 otherwise
        """
        return self.label_to_idx.get(label_str, -1)