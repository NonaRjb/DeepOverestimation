import torch.nn as nn
from collections import OrderedDict


# class MLP(nn.Module):
#     def __init__(self, input_size, hidden_size, n_classes):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, n_classes if n_classes > 2 else 1)

#     def forward(self, x):
#         if len(x.shape) > 2:
#             batch_size = x.shape[0]
#             x = x.view(batch_size, -1)
#         x = x.squeeze(dim=1)
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes, n_layers=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        hidden_list = []
        for l in range(n_layers-1):
            hidden_list.append(('ln' + str(l+1), nn.Linear(hidden_size, hidden_size)))
            hidden_list.append(('bn' + str(l+1), nn.BatchNorm1d(hidden_size)))
            hidden_list.append(('relu' + str(l+1), nn.ReLU()))
        hidden_list.append(('lnout', nn.Linear(hidden_size, n_classes if n_classes > 2 else 1)))
        self.hidden_layers = nn.Sequential(OrderedDict(hidden_list))
    
    def forward(self, x):
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
        x = x.squeeze(dim=1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.hidden_layers(out)
        return out

