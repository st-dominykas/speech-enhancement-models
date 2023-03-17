import os
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import re

def slice_signal(file, window_size, stride, sample_rate):
    """
    Helper function for slicing the audio file
    by window size and sample rate with [1-stride] percent overlap (default 50%).
    """
    wav, _ = torchaudio.load(file)
    wav = torchaudio.transforms.Resample(_,sample_rate)(wav).flatten()
    wav = wav.numpy()
    hop = int(window_size * stride)
    slices = []
    for end_idx in range(window_size, len(wav), hop):
        start_idx = end_idx - window_size
        slice_sig = wav[start_idx:end_idx]
        slices.append(slice_sig)
    return slices

def process_and_serialize(window_size, sample_rate, data_type, df_labels, serialized_folder):
    """
    Serialize, down-sample the sliced signals and save on separate folder.
    """
    stride = 0.5
    clean_files = df_labels['File Name']
    noisy_files = df_labels['Corrupted File Name']

    # walk through the path, slice the audio file, and save the serialized result
    for index in range(len(clean_files)):
        clean_sliced = slice_signal(clean_files[index], window_size, stride, sample_rate)
        noisy_sliced = slice_signal(noisy_files[index], window_size, stride, sample_rate)
        
        file_name = re.split(r'/',clean_files[index])[-1]
        for idx, slice_tuple in enumerate(zip(clean_sliced, noisy_sliced)):
            pair = np.array([slice_tuple[0], slice_tuple[1]])
            np.save(os.path.join(serialized_folder, '{}_{}'.format(file_name, idx)), arr=pair)

class AudioDataset(Dataset):
    """
    Audio sample reader.
    """

    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError('The {} data folder does not exist!')

        self.file_names = [os.path.join(data_path, filename) for filename in os.listdir(data_path)]

    def __getitem__(self, idx):
        pair = np.load(self.file_names[idx])
        noisy = pair[1].reshape(1, -1)
        clean = pair[0].reshape(1, -1)
        return torch.from_numpy(clean).type(torch.FloatTensor), torch.from_numpy(noisy).type(torch.FloatTensor)

    def __len__(self):
        return len(self.file_names)