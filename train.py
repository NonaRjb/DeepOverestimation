from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn as nn
import torch
import argparse
import numpy as np

from dataset.data import RandomDataset
from model.architectures import MLP
from model.trainer import Trainer
import utils


root_path = "/Volumes/T5 EVO/Overfitting/cov_data"


def seed_everything(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('-n', '--n_samples', type=int, default=1000)
    parser.add_argument('-d', '--n_features', type=int, default=2)
    parser.add_argument('--cov_file', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = args.seed
    seed_everything(seed)
    # training vars
    batch_size = args.batch
    epochs = args.epochs
    lr = args.lr
    weight_decay = 0.01
    # data vars
    n = args.n_samples
    d = args.n_features
    cov_file = args.cov_file

    cov_mat = utils.load_cov_mat(root_path, cov_file, d)

    data = RandomDataset(length=n, n_features=d, cov_mat=cov_mat)
    g = torch.Generator().manual_seed(seed)
    outer_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=seed)
    inner_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=seed)

    val_scores = []
    test_scores = []

    for i, (train_all_index, test_index) in enumerate(outer_cv.split(data.samples, data.labels)):
        print(f"##### Test Fold {i} #####\n")
        for j, (train_index, val_index) in enumerate(
                inner_cv.split(data.samples[train_all_index], data.labels[train_all_index])):
            train_index = train_all_index[train_index]
            val_index = train_all_index[val_index]
            train_sub_sampler = torch.utils.data.SubsetRandomSampler(train_index, generator=g)
            val_sub_sampler = torch.utils.data.SubsetRandomSampler(val_index, generator=g)
            test_sub_sampler = torch.utils.data.SubsetRandomSampler(test_index, generator=g)
            # Create DataLoader for training and validation
            train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sub_sampler, drop_last=True)
            val_loader = DataLoader(data, batch_size=batch_size, sampler=val_sub_sampler, drop_last=False)
            test_loader = DataLoader(data, batch_size=batch_size, sampler=test_sub_sampler, drop_last=False)

            model = MLP(input_size=d, hidden_size=32, n_classes=2)
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
            loss = nn.BCEWithLogitsLoss().to(device)
            trainer = Trainer(model, optimizer, loss, epochs, batch_size, device=device)
            best_model = trainer.train(train_loader, val_loader)

            model.load_state_dict(best_model['model_state_dict'])
            test_loss, test_auroc = trainer.evaluate(model, test_loader)
            print(f"Test Loss = {test_loss}, Test ROC-AUC = {test_auroc}\n")

            val_scores.append(best_model['auroc'])
            test_scores.append(test_auroc)

    print(np.mean(val_scores), np.mean(test_scores))
