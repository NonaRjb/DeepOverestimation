import torch
import numpy as np
import argparse
import os


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
    parser.add_argument('-n', '--n_samples', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=1000)
    parser.add_argument('--n_channels', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--experiment', type=str, default='test')
    parser.add_argument('--save_path', type=str, default='/local_storage/datasets/nonar/Overfitting/cov_data')
    parser.add_argument('--root_path', type=str, default='/Midgard/home/nonar/data/Overfitting/')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    root_path = args.root_path
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
    # data vars
    n = args.n_samples
    n_test = args.n_test
    ch = args.n_channels




