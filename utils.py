import numpy as np
import os


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
    else:
        assert ratio < 1
        ratio = int(ratio * d)
    cov_mat[:ratio, :] = 10 * cov_mat[:ratio, :]
    cov_mat[ratio:, :] = 0.01 * cov_mat[ratio:, :]
    return cov_mat


