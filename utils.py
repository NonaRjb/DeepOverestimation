import scipy.io as scio
from scipy.signal import resample
import mat73
import mne
import numpy as np
from scipy.io import loadmat
import os
import pickle

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


def load_real_eeg(root, subject_id, data_type, **kwargs):
    if data_type == 'source':
        filename = "source_data.mat"
        file = os.path.join(root, str(subject_id), filename)
        data, labels, time, fs = load_source_data(file, kwargs['fs_new'])
    elif data_type == "sensor":
        filename = "preprocessed_data.mat"
        file = os.path.join(root, str(subject_id), filename)
        data, labels, time, fs = load_sensor_data(file, kwargs['fs_new'])
    elif data_type == "sensor_ica":
        filename = "preprocessed_data_ica.mat"
        file = os.path.join(root, str(subject_id), filename)
        data, labels, time, fs = load_sensor_ica_data(file, kwargs['fs_new'])
    elif data_type == "sniff":
        filename = "s" + str("{:02d}".format(subject_id)) + "_sniff.pkl"
        file = os.path.join(root, str(subject_id), filename)
        data, labels, time, fs = load_sniff_data(file, kwargs['fs_new'])
    else:
        raise NotImplementedError

    return data, labels, time, fs


def load_sensor_data(filename, fs_new=None):
    print(f"********** loading sensor data from {filename} **********")
    data_struct = mat73.loadmat(filename)
    data = np.asarray(data_struct['data_eeg']['trial'])
    time = data_struct['data_eeg']['time'][0]
    time = np.asarray(time)
    labels = data_struct['data_eeg']['trialinfo'].squeeze()
    labels = labels[:, 0]
    channels = data_struct['data_eeg_ica']['label']
    channels = [ch[0] for ch in channels]
    fs = 512

    if fs_new is None:
        return data, labels, time, fs
    else:
        # low-pass filter
        data_filtered = mne.filter.filter_data(data, sfreq=fs, l_freq=None, h_freq=min(120, fs_new/2))
        times_new = np.arange(-1., 4., 1 / fs_new)
        data_resampled = resample(data_filtered, num=int(fs_new / fs * data_filtered.shape[-1]), axis=-1)
        return data_resampled, labels, times_new, fs_new


def load_sensor_ica_data(filename, fs_new=None):
    print(f"********** loading sensor data from {filename} **********")
    data_struct = mat73.loadmat(filename)
    data = np.asarray(data_struct['data_eeg_ica']['trial'])
    time = data_struct['data_eeg_ica']['time'][0]
    time = np.asarray(time)
    labels = data_struct['data_eeg_ica']['trialinfo'].squeeze()
    labels = labels[:, 0]
    channels = data_struct['data_eeg_ica']['label']
    channels = [ch[0] for ch in channels]
    fs = 512

    if fs_new is None:
        return data, labels, time, fs
    else:
        # low-pass filter
        data_filtered = mne.filter.filter_data(data, sfreq=fs, l_freq=None, h_freq=min(120, fs_new/2))
        times_new = np.arange(-1., 4., 1 / fs_new)
        data_resampled = resample(data_filtered, num=int(fs_new / fs * data_filtered.shape[-1]), axis=-1)
        return data_resampled, labels, times_new, fs_new


def load_source_data(filename, fs_new=None):
    print(f"********** loading source data from {filename} **********")

    mat73_files = [i for i in range(21, 54)]
    mat73_files.append(3)
    print(mat73_files)
    if int(filename.split("/")[-2]) in mat73_files:
        data_struct = mat73.loadmat(filename)

        data = np.asarray(data_struct['source_data_ROI']['trial'])
        time = data_struct['source_data_ROI']['time'][0]
        labels = data_struct['source_data_ROI']['trialinfo'].squeeze()
    else:
        data_struct = scio.loadmat(filename)

        data = np.asarray(list(data_struct['source_data_ROI']['trial'][0][0][0]))
        time = data_struct['source_data_ROI']['time'][0][0][0][0].squeeze()
        labels = data_struct['source_data_ROI']['trialinfo'][0][0].squeeze()
    fs = 512
    labels = labels[:, 0]
    time = np.asarray(time)

    if fs_new is None:
        return data, labels, time, fs
    else:
        # low-pass filter
        data_filtered = mne.filter.filter_data(data, sfreq=fs, l_freq=None, h_freq=min(120, fs_new/2))
        times_new = np.arange(-1., np.max(time), 1 / fs_new)
        data_resampled = resample(data_filtered, num=int(fs_new / fs * data_filtered.shape[-1]), axis=-1)
        return data_resampled, labels, times_new, fs_new


def load_sniff_data(filename, fs_new=None):
    fs = 400
    orig_times = np.arange(-1., 4., 1 / fs)

    with open(filename, "rb") as f:
        data_dict = pickle.load(f)

    trials = data_dict['trials']
    labels = data_dict['labels']

    # Low-pass filter and downsample TODO
    if fs_new is None:
        return trials, labels, orig_times, fs
    else:
        # low-pass filter
        trials_filtered = mne.filter.filter_data(trials, sfreq=fs, l_freq=None, h_freq=50)
        times_new = np.arange(-1., 4., 1 / fs_new)
        trials_resampled = resample(trials_filtered, num=int(fs_new / fs * trials_filtered.shape[-1]), axis=-1)
        return trials_resampled, labels, times_new, fs_new


def load_sniff_mat(path, save_path):
    # Function to load the sniff signal MATLAB struct and split it into per s signals

    labels_map = {
        '1-Butanol low  ': 1,
        '5-Nonanone low ': 2,
        'Undecanal low  ': 4,
        '1-Butanol high ': 8,
        '5-Nonanone high': 16,
        'Undecanal high ': 32,
        'Air            ': 64
    }
    data_struct = scio.loadmat(path)
    data_all = data_struct['data'][0][0]
    for s in range(1, 54):
        data_subject = data_all['subject' + str("{:02d}".format(s))][0]
        trials_subject = data_subject['trials'][0]
        labels_subject = data_subject['odors'][0].squeeze()
        labels_subject = [labels_map[i[0]] for i in labels_subject]
        data_dict = {
            'trials': trials_subject,
            'labels': labels_subject
        }
        with open(os.path.join(save_path, "s" + str("{:02d}".format(s)) + "_sniff.pkl"), "wb") as f:
            pickle.dump(data_dict, f)
    return


def crop_temporal(data, tmin, tmax, tvec, w=None, fs=512):
    if tmin is None:
        t_min = 0
    else:
        t_min = np.abs(tvec - tmin).argmin()

    if w is None:
        if tmax is None:
            t_max = len(tvec)
        else:
            t_max = np.abs(tvec - tmax).argmin()
    else:
        if tmin is None:
            t_max = int(w * fs)
        else:
            tmax = tmin + w
            t_max = np.abs(tvec - tmax).argmin()
    return data[..., t_min:t_max]


def crop_tfr(tfr, tmin, tmax, fmin, fmax, tvec, freqs, w=None, fs=512) -> np.ndarray:
    """
    :param tfr: 4-d data array with the shape (n_trials, n_channels, n_freqs, n_samples)
    :return: cropped array
    """

    if tmin is None:
        t_min = 0
    else:
        t_min = np.abs(tvec - tmin).argmin()

    if w is None:
        if tmax is None:
            t_max = len(tvec)
        else:
            t_max = np.abs(tvec - tmax).argmin()
    else:
        if tmin is None:
            t_max = int(w * fs)
        else:
            tmax = tmin + w
            t_max = np.abs(tvec - tmax).argmin()

    # print(f"Number of Time Samples: {t_max - t_min}")

    if fmin is None:
        f_min = 0
    else:
        f_min = np.abs(freqs - fmin).argmin()
    if fmax is None:
        f_max = len(freqs)
    else:
        f_max = np.abs(freqs - fmax).argmin()

    return tfr[:, :, f_min:f_max, t_min:t_max]


def apply_baseline(tfr, bl_lim, tvec, mode):
    if bl_lim[0] is None:
        baseline_min = 0
    else:
        baseline_min = np.abs(tvec - bl_lim[0]).argmin()
    if bl_lim[1] is None:
        baseline_max = len(tvec)
    else:
        baseline_max = np.abs(tvec - bl_lim[1]).argmin()

    baseline = np.mean(tfr[..., baseline_min:baseline_max], axis=-1, keepdims=True)

    if mode == "mean":
        def fun(d, m):
            d -= m
    elif mode == "ratio":
        def fun(d, m):
            d /= m
    elif mode == "logratio":
        def fun(d, m):
            d /= m
            np.log10(d, out=d)
    elif mode == "percent":
        def fun(d, m):
            d -= m
            d /= m
    elif mode == "zscore":
        def fun(d, m):
            d -= m
            d /= np.std(d[..., baseline_min:baseline_max], axis=-1, keepdims=True)
    elif mode == "zlogratio":
        def fun(d, m):
            d /= m
            np.log10(d, out=d)
            d /= np.std(d[..., baseline_min:baseline_max], axis=-1, keepdims=True)
    else:
        raise NotImplementedError

    fun(tfr, baseline)
    return tfr


def get_n_channels(modality):
    if modality == "source":
        n_channels = 4
    elif modality == "ebg":
        n_channels = 4
    elif modality == "eeg":
        n_channels = 63
    elif modality in ["ebg-sniff", "sniff-ebg"]:
        n_channels = 5
    elif modality in ["eeg-sniff", "sniff-eeg"]:
        n_channels = 64
    elif modality == "both-sniff":
        n_channels = 68
    elif modality == 'sniff':
        n_channels = 1
    elif modality in ['source-sniff', 'sniff-source']:
        n_channels = 5
    elif modality in ['source-ebg', 'ebg-source']:
        n_channels = 8
    elif modality in ['source-eeg', 'eeg-source']:
        n_channels = 67
    elif modality in ['eeg-ebg', 'ebg-eeg']:
        n_channels = 67
    else:
        raise NotImplementedError
    return n_channels


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
        return architectures.ConvNet(
            n_channels=kwargs['n_channels'], 
            n_samples=kwargs['n_samples'], 
            n_classes=kwargs['n_classes'])
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
