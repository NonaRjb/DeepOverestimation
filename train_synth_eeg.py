import os
import pickle
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adam, SGD
import torch.nn as nn
import torch
import argparse
import numpy as np
import warnings

from dataset.data import Synthetic_EEG
from model.architectures import ConvNet
from model.trainer import Trainer
import utils


def seed_everything(seed_val):
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optim', type=str, default='adamw')
    parser.add_argument('--model', type=str, default="convnet")
    parser.add_argument('-n', '--n_samples', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--n_channels', type=int, default=32)
    parser.add_argument('--frame_size', type=int, default=125)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment', type=str, default='test')
    parser.add_argument('--save_path', type=str, default='/Midgard/home/nonar/data/Overfitting/')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, args.experiment)
    os.makedirs(save_path, exist_ok=True)

    seed = args.seed
    test_offset = 147
    seed_everything(seed)
    # training vars
    batch_size = args.batch
    epochs = args.epochs
    optim_name = args.optim
    lr = args.lr
    weight_decay = 0.
    # architecture
    model_name = args.model
    # data vars
    n = args.n_samples
    n_test = args.n_test
    ch = args.n_channels
    frame = args.frame_size  # TODO: currently only works with size 128 (because the linear layer input size that is hard-coded)

    save_dir = "n" + str(n) + "_ch" + str(ch) + "_f" + str(frame) + "_" + model_name
    save_path = os.path.join(save_path, save_dir)
    os.makedirs(save_path, exist_ok=True)

    test_data = Synthetic_EEG(n_trials=n_test, n_channels=ch, n_samples=frame, fs=256, seed=seed+test_offset)
    data = Synthetic_EEG(n_trials=n, n_channels=ch, n_samples=frame, fs=256, seed=seed)
    g = torch.Generator().manual_seed(seed)
    inner_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=seed)

    val_scores = []
    test_scores = []
    train_scores = []
    val_losses = []
    test_losses = []
    best_epochs = []

    for j, (train_index, val_index) in enumerate(
            inner_cv.split(data.labels, data.labels)):
        
        train_sub_sampler = torch.utils.data.SubsetRandomSampler(train_index, generator=g)
        val_sub_sampler = torch.utils.data.SubsetRandomSampler(val_index, generator=g)
    
        # Create DataLoader for training, test and validation
        train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sub_sampler, drop_last=True)
        val_loader = DataLoader(data, batch_size=batch_size, sampler=val_sub_sampler, drop_last=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, drop_last=False)

        model = utils.load_model(model_name=model_name, n_channels=ch, n_samples=frame, n_classes=2)
        # model = ConvNet(n_channels=ch) # TODO: softmax in the model and sigmoid in the trainer (currently commented softmax)
        if optim_name == "adamw":
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        elif optim_name == "adam":
            optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim_name == "sgd":
            optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise NotImplementedError
        loss = nn.BCEWithLogitsLoss().to(device)
        trainer = Trainer(model, optimizer, loss, epochs, batch_size, device=device)
        best_model = trainer.train(train_loader, val_loader)
        model.load_state_dict(best_model['model_state_dict'])
        test_loss, test_auroc, _, _ = trainer.evaluate(model, test_loader)
        print(f"Test Loss = {test_loss}, Test ROC-AUC = {test_auroc}\n")
        train_scores.append(best_model['train_auroc'].numpy())
        val_scores.append(best_model['auroc'].numpy())
        test_scores.append(test_auroc.numpy())
        val_losses.append(best_model['loss'])
        test_losses.append(test_loss)
        best_epochs.append(best_model['epoch'])

    scores = {
        'train': np.asarray(train_scores), 
        'val': np.asarray(val_scores), 
        'test': np.asarray(test_scores), 
        'val_loss': np.asarray(val_losses),
        'test_loss': np.asarray(test_losses),
        'epoch': best_epochs}
    with open(os.path.join(save_path, "scores.pkl"), 'wb') as f:
        pickle.dump(scores, f)
    print(np.mean(val_scores), np.mean(test_scores))






