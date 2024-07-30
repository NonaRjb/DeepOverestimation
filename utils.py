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
