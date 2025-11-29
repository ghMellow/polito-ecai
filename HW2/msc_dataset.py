import os
import torch
import soundfile as sf
from torch.utils.data import Dataset


class MSCDataset(Dataset):
    """
    Custom Dataset class for Mini Speech Commands.
    """
    
    def __init__(self, root, classes, split='training', preprocess=None, download=False):
        """
        Initialize the MSC Dataset with specified classes and split.
        """
        self.root = root
        self.classes = classes
        self.split = split
        self.preprocess = preprocess
        self.label_to_idx = {label: idx for idx, label in enumerate(classes)}
        
        if download:
            self._download()
        
        split_mapping = {
            'training': 'msc-training',
            'validation': 'msc-validation',
            'testing': 'msc-testing'
        }
        
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
        
        self.samples = []
        
        if os.path.exists(data_folder) and os.path.isdir(data_folder):
            for filename in os.listdir(data_folder):
                if filename.endswith('.wav'):
                    label = filename.split('_')[0]
                    if label in classes:
                        file_path = os.path.join(data_folder, filename)
                        self.samples.append((file_path, label))
        
        print(f"Loaded {len(self.samples)} samples from {data_folder} for classes {classes}")
    
    def _download(self):
        """
        Placeholder for dataset download functionality.
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
        Returns a dictionary with audio tensor 'x' and integer label 'y'.
        """
        file_path, label_str = self.samples[idx]
        
        waveform, sampling_rate = sf.read(file_path, dtype='float32')
        waveform = torch.from_numpy(waveform).unsqueeze(0)
        
        target_length = 16000
        if waveform.shape[1] < target_length:
            pad_length = target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        elif waveform.shape[1] > target_length:
            waveform = waveform[:, :target_length]
        
        if self.preprocess is not None:
            waveform = self.preprocess(waveform)
        
        label_int = self.label_to_idx[label_str]
        
        return {
            "x": waveform,
            "y": label_int
        }
    
    def label_to_int(self, label_str):
        """
        Convert a label string to its corresponding integer index.
        Returns the integer index corresponding to the label, -1 otherwise.
        """
        return self.label_to_idx.get(label_str, -1)