import torch
import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes, n_layers=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        hidden_list = []
        for l in range(n_layers - 1):
            hidden_list.append(('ln' + str(l + 1), nn.Linear(hidden_size, hidden_size)))
            hidden_list.append(('bn' + str(l + 1), nn.BatchNorm1d(hidden_size)))
            hidden_list.append(('relu' + str(l + 1), nn.ReLU()))
        hidden_list.append(('lnout', nn.Linear(hidden_size, n_classes if n_classes > 2 else 1)))
        self.hidden_layers = nn.Sequential(OrderedDict(hidden_list))

    def forward(self, x):
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
        x = x.squeeze(dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.hidden_layers(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, n_channels, n_classes=2):
        super(ConvNet, self).__init__()

        self.temporal_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 25)),
            nn.BatchNorm2d(40)
        )
        self.spatial_block = nn.Sequential(
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(n_channels, 1)),
            nn.BatchNorm2d(40)
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 15), stride=(1, 3))
        self.fc = nn.Linear(in_features=1160, out_features=n_classes if n_classes > 2 else 1)
        self.activation = nn.Softmax()

    def forward(self, x):
        x = self.temporal_block(x)
        x = self.spatial_block(x)
        x = x.squeeze(2)
        x = self.avg_pool(x)

        # Flatten the tensor, keeping the batch size
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return self.activation(x)


if __name__ == "__main__":
    # Calculate in_features
    channels = 32  # Example value, adjust according to your actual data
    n_samples = 125  # Example value, adjust according to your actual data
    classes = 2  # Example value, adjust according to your actual case

    # Instantiate the model
    model = ConvNet(channels, classes)

    # Create a dummy input with the shape (batch_size, n_channels, n_samples)
    dummy_input = torch.randn(1, 1, channels, n_samples)

    # Pass through the model up to the flattening stage
    with torch.no_grad():
        out = model.temporal_block(dummy_input)
        out = model.spatial_block(out)
        out = out.squeeze(2)
        out = model.avg_pool(out)
        out = out.view(out.size(0), -1)  # Flatten

    # The shape of 'out' now should be (batch_size, in_features)
    in_features = out.size(1)

    # Now set the in_features of the fully connected layer
    model.fc = nn.Linear(in_features=in_features, out_features=classes if classes > 2 else 1)

    print(f'in_features of the Linear layer: {in_features}')
