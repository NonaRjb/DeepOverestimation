import numpy as np
import torch
from torch.utils.data import Dataset
import math

from utils import generate_noise_eeg, load_real_eeg


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
            print(f"generate data for channel {ch}")
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
        sample = sample.unsqueeze(0)
        label = self.labels[idx]

        return sample, label


class Real_EEG(Dataset):
    def __init__(
            self, root_path: str,
            source_data,
            label,
            time_vec,
            fs,
            tmin: float = None,
            tmax: float = None,
            data_type: str = 'source',
            modality: str = 'both',
            pick_subjects: int = 0,
            normalize: bool = False,
            fs_new: int = None,
            transform=None
    ):

        self.root_path = root_path

        self.baseline_min = -1.0
        self.baseline_max = -0.6
        self.normalize = normalize
        self.source_data = None
        self.labels = None
        self.subject_id = None
        self.fs = float(fs)
        self.time_vec = time_vec
        self.class_weight = None
        self.modality = modality
        self.transform = transform

        subject = pick_subjects
        # for i, subject in enumerate(subjects):
        # source_data, label, time_vec, fs = load_ebg4(root_path, subject, data_type, fs_new=fs_new)

        if data_type == 'sensor' or data_type == 'sensor_ica':
            if modality == 'eeg':
                source_data = source_data[:, :63, :]
            elif modality == 'eeg-sniff' or modality == 'sniff-eeg':
                sniff_data, _, _, _ = load_real_eeg(
                    root_path,
                    subject,
                    data_type="sniff",
                    fs_new=fs_new if fs_new is not None else fs
                )
                sniff_data = np.expand_dims(sniff_data, axis=1)
                source_data = source_data[:, :63, :]  # extract EEG
                source_data = np.concatenate((source_data, sniff_data), axis=1)
            elif modality == 'ebg':
                source_data = source_data[:, 63:-1, :]
            elif modality == 'ebg-sniff' or modality == 'sniff-ebg':
                sniff_data, _, _, _ = load_real_eeg(
                    root_path,
                    subject,
                    data_type="sniff",
                    fs_new=fs_new if fs_new is not None else fs
                )
                sniff_data = np.expand_dims(sniff_data, axis=1)
                source_data = source_data[:, 63:-1, :]  # extract EBG
                source_data = np.concatenate((source_data, sniff_data), axis=1)
            elif modality == 'eeg-ebg' or modality == 'ebg-eeg':
                source_data = source_data[:, :-1, :]
            elif modality == 'both-sniff':
                sniff_data, _, _, _ = load_real_eeg(
                    root_path,
                    subject,
                    data_type="sniff",
                    fs_new=fs_new if fs_new is not None else fs
                )
                sniff_data = np.expand_dims(sniff_data, axis=1)
                source_data = np.concatenate((source_data[:, :-1, :], sniff_data), axis=1)
            elif modality == 'sniff':
                sniff_data, _, _, _ = load_real_eeg(
                    root_path,
                    subject,
                    data_type="sniff",
                    fs_new=fs_new if fs_new is not None else fs
                )
                sniff_data = np.expand_dims(sniff_data, axis=1)
                source_data = sniff_data
            elif modality == 'source-ebg' or modality == 'ebg-source':
                source_activity, _, t_source, _ = load_real_eeg(root_path, subject, "source", fs_new=fs_new)
                time_vec = t_source
                source_data = source_data[:, 63:-1, :len(t_source)]
                source_data = np.concatenate((source_data, source_activity), axis=1)
            elif modality == 'source-eeg' or modality == 'eeg-source':
                source_activity, _, t_source, _ = load_real_eeg(root_path, subject, "source", fs_new=fs_new)
                time_vec = t_source
                source_data = source_data[:, :63, :len(t_source)]
                source_data = np.concatenate((source_data, source_activity), axis=1)
            else:
                raise NotImplementedError
        else:  # data type is source
            if modality == 'source-sniff' or modality == 'sniff-source':
                sniff_data, _, _, _ = load_real_eeg(
                    root_path,
                    subject,
                    data_type="sniff",
                    fs_new=fs_new if fs_new is not None else fs
                )
                sniff_data = sniff_data[..., :len(time_vec)]
                sniff_data = np.expand_dims(sniff_data, axis=1)
                source_data = np.concatenate((source_data, sniff_data), axis=1)
            elif modality == 'source-ebg' or modality == 'ebg-source':
                sensor_data, _, _, _ = load_real_eeg(root_path, subject, "sensor_ica", fs_new=fs_new)
                sensor_data = sensor_data[:, 63:-1, :len(time_vec)]
                source_data = np.concatenate((sensor_data, source_data), axis=1)
            elif modality == 'source-eeg' or modality == 'eeg-source':
                sensor_data, _, _, _ = load_real_eeg(root_path, subject, "sensor_ica", fs_new=fs_new)
                sensor_data = sensor_data[:, :63, :len(time_vec)]
                source_data = np.concatenate((sensor_data, source_data), axis=1)
            else:
                pass

        if self.source_data is None:
            self.source_data = source_data
            self.labels = np.expand_dims(label, axis=1)
            self.subject_id = subject * np.ones((len(label), 1))
        else:
            self.source_data = np.vstack((self.source_data, source_data))
            self.labels = np.vstack((self.labels, np.expand_dims(label, axis=1)))
            self.subject_id = np.vstack((self.subject_id, self.subject_id * np.ones((len(label), 1))))
        if tmin is None:
            self.t_min = 0
        else:
            self.t_min = np.abs(self.time_vec - tmin).argmin()

        if tmax is None:
            self.t_max = len(self.time_vec)
        else:
            self.t_max = np.abs(self.time_vec - tmax).argmin()

        print(f"first time sample: {self.t_min}, last time sample: {self.t_max}")

        self.baseline_min = np.abs(self.time_vec - self.baseline_min).argmin()
        self.baseline_max = np.abs(self.time_vec - self.baseline_max).argmin()
        self.time_vec = self.time_vec[self.t_min:self.t_max]

        # only consider high intensity odors
        mask = np.logical_not(np.isin(self.labels.squeeze(), [1, 2, 4]))
        self.source_data = self.source_data[mask, ...]
        self.labels = self.labels[mask]
        new_labels = [1. if y == 64 else 0. for y in self.labels]
        self.labels = new_labels
        class_0_count = new_labels.count(0.)
        class_1_count = new_labels.count(1.)
        print(f"N(class 0) = {class_0_count}, N(class 1) = {class_1_count}")
        self.class_weight = torch.tensor(class_0_count / class_1_count)

        self.data = self.source_data
        self.baseline = np.mean(self.data[..., self.baseline_min:self.baseline_max], axis=(0, -1), keepdims=True)
        # self.baseline = np.mean(self.data[..., self.t_min:self.t_max], axis=-1, keepdims=True)
        self.data = self.data[..., self.t_min:self.t_max] - self.baseline
        self.percentile_95 = np.percentile(np.abs(self.data), 95, axis=-1, keepdims=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()
        sample = self.data[item, ...]
        if self.normalize or self.modality == "sniff":
            sample = sample / self.percentile_95[item, ...]
        elif self.modality == 'ebg-sniff' or self.modality == 'eeg-sniff':
            sample_normalized = sample / self.percentile_95[item, ...]
            sample[-1, ...] = sample_normalized[-1, ...]

        sample = torch.from_numpy(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample, self.labels[item]

