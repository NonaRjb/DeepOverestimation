import os
import pickle

from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import torch
import argparse
import numpy as np
import warnings

from dataset.data import RandomDataset
from model.architectures import MLP
from model.trainer import Trainer
import utils


local_root_path = "/Volumes/T5 EVO/Overfitting/cov_data"
local_save_path = "/Volumes/T5 EVO/Overfitting/out/"
remote_root_path = "/local_storage/datasets/nonar/Overfitting/cov_data"
remote_save_path = "/Midgard/home/nonar/data/Overfitting/"


def seed_everything(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=8)
    parser.add_argument('-n', '--n_samples', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('-d', '--n_features', type=int, default=2)
    parser.add_argument('-r', '--n_large', type=float, default=1)
    parser.add_argument('--cov_file', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment', type=str, default='test')
    return parser.parse_args()


if __name__ == "__main__":
    remote = True

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_path = remote_root_path if remote else local_root_path
    save_path = remote_save_path if remote else local_save_path
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, args.experiment)
    os.makedirs(save_path, exist_ok=True)

    seed = args.seed
    test_offset = 147
    seed_everything(seed)
    # training vars
    batch_size = args.batch
    epochs = args.epochs
    lr = args.lr
    weight_decay = 0.01
    hidden_size = args.hidden_size
    # data vars
    n = args.n_samples
    n_test = args.n_test
    d = args.n_features
    r = args.n_large
    cov_file = args.cov_file

    if cov_file is not None:
        cov_mat = utils.load_cov_mat(root_path, cov_file, d)
        assert cov_mat.shape[0] == cov_mat.shape[1]
        if cov_mat.shape[0] != d:
            warnings.warn(f"Number of input features, d, "
                        f"differ from the covariance matrix dimensions. d is set to {cov_mat.shape[0]}")
            d = cov_mat.shape[0]
        save_dir = cov_file.split(".")[0] + "_out"
    else:
        cov_mat = utils.generate_cov_mat(d=d, ratio=r)
        save_dir = str(d) + "d_" + str(r)

    save_path = os.path.join(save_path, save_dir)
    os.makedirs(save_path, exist_ok=True)

    test_data = RandomDataset(length=n_test, n_features=d, cov_mat=cov_mat, seed=seed+test_offset)
    data = RandomDataset(length=n, n_features=d, cov_mat=cov_mat, seed=seed)
    g = torch.Generator().manual_seed(seed)
    # outer_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=seed)
    inner_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=seed)

    val_scores = []
    test_scores = []
    train_scores = []
    best_epochs = []

    for j, (train_index, val_index) in enumerate(
            inner_cv.split(data.labels, data.labels)):
        
        train_sub_sampler = torch.utils.data.SubsetRandomSampler(train_index, generator=g)
        val_sub_sampler = torch.utils.data.SubsetRandomSampler(val_index, generator=g)
    
        # Create DataLoader for training, test and validation
        train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sub_sampler, drop_last=True)
        val_loader = DataLoader(data, batch_size=batch_size, sampler=val_sub_sampler, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, drop_last=False)

        model = MLP(input_size=d, hidden_size=hidden_size, n_classes=2)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        loss = nn.BCEWithLogitsLoss().to(device)
        trainer = Trainer(model, optimizer, loss, epochs, batch_size, device=device)
        best_model = trainer.train(train_loader, val_loader)
        model.load_state_dict(best_model['model_state_dict'])
        test_loss, test_auroc = trainer.evaluate(model, test_loader)
        print(f"Test Loss = {test_loss}, Test ROC-AUC = {test_auroc}\n")
        train_scores.append(best_model['train_auroc'].numpy())
        val_scores.append(best_model['auroc'].numpy())
        test_scores.append(test_auroc.numpy())
        best_epochs.append(best_model['epoch'])

    scores = {'train': np.asarray(train_scores), 'val': np.asarray(val_scores), 'test': np.asarray(test_scores), 'epoch': best_epochs}
    with open(os.path.join(save_path, "scores.pkl"), 'wb') as f:
        pickle.dump(scores, f)
    print(np.mean(val_scores), np.mean(test_scores))
