import numpy as np
import torch
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    def __init__(self, length: int, n_features: int, cov_mat: np.ndarray):
        """
    Args:
        length (int): Number of samples in the dataset.
        n_features (int): Data dimensionality (or umber of features).
        cov_mat (array):
    """

        self.length = length
        self.shape = n_features
        # self.samples = torch.randn((self.length,)+self.shape)
        self.samples = np.random.multivariate_normal(mean=np.zeros((n_features,)), cov=cov_mat, size=length)
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
        # sample = self.samples[idx]
        sample = torch.from_numpy(self.samples[idx]).float()
        label = self.labels[idx]

        return sample, label
