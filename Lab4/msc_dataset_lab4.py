import os
import torch
import torchaudio
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
        
        # Map split names to actual folder names
        split_mapping = {
            'training': 'msc-train',
            'validation': 'msc-val',
            'testing': 'msc-test'
        }
        
        # Determine the data folder based on split
        if split in split_mapping:
            # Try /tmp/ first (Deepnote absolute location), then relative tmp/
            abs_tmp_folder = os.path.join('/tmp', split_mapping[split])
            rel_tmp_folder = os.path.join(root, 'tmp', split_mapping[split])
            root_folder = os.path.join(root, split_mapping[split])
            
            # DEBUG: Print what we're checking
            print(f"Checking paths for split '{split}':")
            print(f"  {abs_tmp_folder} -> exists: {os.path.exists(abs_tmp_folder)}")
            print(f"  {rel_tmp_folder} -> exists: {os.path.exists(rel_tmp_folder)}")
            print(f"  {root_folder} -> exists: {os.path.exists(root_folder)}")
            
            if os.path.exists(abs_tmp_folder):
                data_folder = abs_tmp_folder
            elif os.path.exists(rel_tmp_folder):
                data_folder = rel_tmp_folder
            elif os.path.exists(root_folder):
                data_folder = root_folder
            else:
                data_folder = root
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
                
        # Load audio
        waveform, sampling_rate = torchaudio.load(file_path)
        
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