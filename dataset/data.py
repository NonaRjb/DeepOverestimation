import numpy as np
import torch
from torch.utils.data import Dataset

from utils import generate_noise_eeg


class RandomDataset(Dataset):
    def __init__(self, length: int, n_features: int, cov_mat: np.ndarray, seed: int = None):
        """
    Args:
        length (int): Number of samples in the dataset.
        n_features (int): Data dimensionality (or umber of features).
        cov_mat (array):
    """
        if seed is not None:
            np.random.seed(seed)

        self.length = length
        self.samples = np.random.multivariate_normal(mean=np.zeros((n_features,)), cov=cov_mat, size=self.length)
        self.labels = np.random.choice([0, 1], size=self.length, p=[0.5, 0.5])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample.

        Returns:
            sample (torch.Tensor): Randomly generated sample.
        """
        # if self.seed is not None:
        #     np.random.seed(self.seed + idx)

        # sample = np.random.multivariate_normal(mean=np.zeros((self.n_features,)), cov=self.cov_mat)
        # sample = torch.from_numpy(sample).float()
        sample = torch.from_numpy(self.samples[idx]).float()
        label = self.labels[idx]

        return sample, label


class Synthetic_EEG(Dataset):
    def __init__(self, n_trials: int, n_channels: int, n_samples: int, fs: int = 250, noise_amp: float = 10,
                 seed: int = None):

        if seed is not None:
            np.random.seed(seed)

        self.n_trials = n_trials
        self.n_channels = n_channels
        self.n_samples = n_samples

        data = np.zeros((self.n_channels, self.n_trials*self.n_samples))
        for ch in range(self.n_channels):
            data[ch, :] += noise_amp * generate_noise_eeg(self.n_samples, self.n_trials, fs)

        temp_data = np.reshape(data, (self.n_channels, self.n_samples, self.n_trials))
        reshaped_data = np.transpose(temp_data, (2, 0, 1))

        self.samples = reshaped_data
        self.labels = np.random.choice([0, 1], size=self.n_trials, p=[0.5, 0.5])

    def __len__(self):
        return self.n_trials

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample.

        Returns:
            sample (torch.Tensor): Randomly generated sample.
        """
        sample = torch.from_numpy(self.samples[idx, ...]).float()
        label = self.labels[idx]

        return sample, label
