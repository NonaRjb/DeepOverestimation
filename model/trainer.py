import matplotlib.pyplot as plt
from torchmetrics import AUROC
from tqdm import tqdm
import numpy as np
import torch
import copy


class Trainer:
    def __init__(
            self, model, optimizer, loss, n_epochs, batch_size, n_classes=2,
            device='cuda'
    ):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss = loss

        self.epochs = n_epochs
        self.batch_size = batch_size

        self.auroc = AUROC(task="binary", num_classes=n_classes)

        # Initialize history dictionaries
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auroc': [],
            'val_auroc': []
        }

    def train(self, train_data_loader, val_data_loader):

        best_model = None
        best_loss = 10000000
        best_acc = 0.0
        for epoch in range(self.epochs):

            self.model.train()
            loss_epoch = []
            y_true = []
            y_pred = []
            progress_bar = tqdm(train_data_loader, disable=True)
            for x, y in progress_bar:
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)

                preds = self.model(x)
                if len(preds.shape) > 1:
                    preds = preds.squeeze(dim=1)
                loss_val = self.loss(preds, y.float())
                loss_epoch.append(loss_val.item())
                y_true.extend(y)
                y_pred.extend(torch.sigmoid(preds))

                loss_val.backward()
                self.optimizer.step()

            with torch.no_grad():
                train_loss = np.mean(loss_epoch)
                train_auroc = self.auroc(torch.stack(y_pred).detach().cpu().float(),
                                         torch.stack(y_true).detach().cpu())

            val_loss, val_auroc = self.evaluate(self.model, val_data_loader)

            # Store metrics in history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auroc'].append(train_auroc)
            self.history['val_auroc'].append(val_auroc)

            if val_loss < best_loss:
                # if val_auroc > best_acc:
                best_loss = val_loss
                # best_acc = val_auroc
                best_model = {
                    'epoch': epoch,
                    'model_state_dict': copy.deepcopy(self.model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                    'loss': val_loss,
                    'auroc': val_auroc
                }

        if best_model is None:
            best_model = {
                'epoch': epoch,
                'model_state_dict': copy.deepcopy(self.model.state_dict()),
                'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
                'loss': val_loss,
                'auroc': val_auroc
            }

        print(f"Train Loss = {train_loss}, Train ROC-AUC = {train_auroc}")
        print(f"Val Loss = {best_model['loss']}, Val ROC-AUC = {best_model['auroc']}")
        # self.plot_history()
        return best_model

    def evaluate(self, model, data_loader):
        model.eval()
        loss_epoch = []
        y_true = []
        y_pred = []
        progress_bar = tqdm(data_loader, disable=True)
        with torch.no_grad():
            for x, y in progress_bar:
                x = x.to(self.device)
                y = y.to(self.device)
                preds = model(x)
                if len(preds.shape) > 1:
                    preds = preds.squeeze(dim=1)
                loss_val = self.loss(preds, y.float())
                loss_epoch.append(loss_val.item())
                y_true.extend(y)
                y_pred.extend(preds)
            mean_loss_epoch = np.mean(loss_epoch)
            auroc = self.auroc(torch.stack(y_pred).detach().cpu().float(),
                               torch.stack(y_true).detach().cpu())
        return mean_loss_epoch, auroc

    def plot_history(self):

        epochs = range(1, self.epochs + 1)

        plt.figure(figsize=(6, 3))

        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], label='Training Loss')
        plt.plot(epochs, self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot training and validation AUROC
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_auroc'], label='Training AUROC')
        plt.plot(epochs, self.history['val_auroc'], label='Validation AUROC')
        plt.xlabel('Epochs')
        plt.ylabel('AUROC')
        plt.title('Training and Validation AUROC')
        plt.legend()

        plt.tight_layout()
        plt.show()
