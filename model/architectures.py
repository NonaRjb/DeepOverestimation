import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, n_classes if n_classes > 2 else 1)

    def forward(self, x):
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
        x = x.squeeze(dim=1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
