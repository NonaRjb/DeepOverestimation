import numpy as np
from scipy.io import loadmat
import os

import model.architectures as architectures


def load_cov_mat(root_path, filename, d):
    if filename is None:
        cov_mat = np.eye(d)
    else:
        file_path = os.path.join(root_path, filename)
        if filename.split('.')[-1] == 'npy':
            cov_mat = np.load(file_path)
        elif filename.split('.')[-1] == 'txt':
            cov_mat = np.loadtxt(file_path, delimiter=',')
        else:
            raise NotImplementedError

    return cov_mat


def generate_cov_mat(d, ratio):
    """

    :param d: feature dimensionality
    :param ratio: number of large eigenvalues or ratio of large eigenvalues to d
    :return:
    """
    cov_mat = np.eye(d)
    if ratio % 1 == 0:
        assert ratio <= d
        ratio = int(ratio)
    else:
        assert ratio < 1
        ratio = int(ratio * d)
    cov_mat[:ratio, :] = 10 * cov_mat[:ratio, :]
    cov_mat[ratio:, :] = 0.01 * cov_mat[ratio:, :]

    return cov_mat


def generate_noise_eeg(frames, epochs, fs):
    """
    Generates noise with the power spectrum of human EEG.

    Parameters:
    frames (int): Number of signal frames per each trial.
    epochs (int): Number of simulated trials.
    fs (float): Sampling rate of simulated signal.

    Returns:
    np.ndarray: Simulated EEG signal; 1D array of size frames * epochs containing concatenated trials.
    """
    # Load the meanpower variable from the .mat file
    mat_contents = loadmat('./dataset/meanpower.mat')
    meanpower = mat_contents['meanpower'].flatten()  # Assuming meanpower is a 1D array

    sumsig = 50  # Number of sinusoids from which each simulated signal is composed

    signal = np.zeros(epochs * frames)
    for trial in range(epochs):
        freq = 0
        range_start = trial * frames
        range_end = (trial + 1) * frames
        for i in range(sumsig):
            freq += 4 * np.random.rand(1)
            freq_index = min(int(np.ceil(freq)), 125) - 1  # MATLAB indexing starts at 1
            freqamp = meanpower[freq_index] / meanpower[0]
            phase = np.random.rand(1) * 2 * np.pi
            signal[range_start:range_end] += np.sin(
                np.arange(1, frames + 1) / fs * 2 * np.pi * freq + phase) * freqamp

    return signal


def load_model(model_name, **kwargs):
    if model_name == "convnet":
        return architectures.ConvNet(n_channels=kwargs['n_channels'], n_classes=kwargs['n_classes'])
    elif model_name == "resnet1d":
        return architectures.ResNet1d(
            n_channels=kwargs["n_channels"],
            n_samples=kwargs["n_samples"],
            net_filter_size=[64, 128, 196, 256, 320],
            net_seq_length=[kwargs['n_samples'], 128, 64, 32, 16],
            n_classes=kwargs["n_classes"],
            kernel_size=17,
            dropout_rate=0.5
        )
    else:
        raise NotImplementedError
