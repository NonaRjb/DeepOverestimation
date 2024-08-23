import torch
import torch.nn as nn
import numpy as np
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
        self.fc = nn.Linear(in_features=1200, out_features=n_classes if n_classes > 2 else 1)
        self.activation = nn.Softmax()

    def forward(self, x):
        x = self.temporal_block(x)
        x = self.spatial_block(x)
        x = x.squeeze(2)
        x = self.avg_pool(x)

        # Flatten the tensor, keeping the batch size
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        # return self.activation(x)
        return x


# Code from GitHub repository of: Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al.
# Deep neural network-estimated electrocardiographic age as a
# mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7.
def _padding(downsample, kernel_size):
    """Compute required padding"""
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding


# Code from GitHub repository of: Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al.
# Deep neural network-estimated electrocardiographic age as a
# mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7.
def _downsample(n_samples_in, n_samples_out):
    """Compute downsample rate"""
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Number of samples for two consecutive blocks "
                         "should always decrease by an integer factor.")
    return downsample


# Code from GitHub repository of: Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al.
# Deep neural network-estimated electrocardiographic age as a
# mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7.
class ResBlock1d(nn.Module):
    """Residual network unit for unidimensional signals."""

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        if kernel_size % 2 == 0:
            raise ValueError("The current implementation only support odd values for `kernel_size`.")
        super(ResBlock1d, self).__init__()
        # Forward path
        padding = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        padding = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                               stride=downsample, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Skip connection
        skip_connection_layers = []
        # Deal with downsampling
        if downsample > 1:
            maxpool = nn.MaxPool1d(downsample, stride=downsample)
            skip_connection_layers += [maxpool]
        # Deal with n_filters dimension increase
        if n_filters_in != n_filters_out:
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False)
            skip_connection_layers += [conv1x1]
        # Build skip conection layer
        if skip_connection_layers:
            self.skip_connection = nn.Sequential(*skip_connection_layers)
        else:
            self.skip_connection = None

    def forward(self, x, y):
        """Residual unit."""
        if self.skip_connection is not None:
            y = self.skip_connection(y)
        else:
            y = y
        # 1st layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # 2nd layer
        x = self.conv2(x)
        x += y  # Sum skip connection and main connection
        y = x
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        return x, y


# Code from GitHub repository of: Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al.
# Deep neural network-estimated electrocardiographic age as a
# mortality predictor. Nat Commun 12, 5117 (2021). https://doi.org/10.1038/s41467-021-25351-7.
class ResNet1d(nn.Module):
    """Residual network for unidimensional signals.
    Parameters
    ----------
    input_dim : tuple
        Input dimensions. Tuple containing dimensions for the neural network
        input tensor. Should be like: ``(n_filters, n_samples)``.
    blocks_dim : list of tuples
        Dimensions of residual blocks.  The i-th tuple should contain the dimensions
        of the output (i-1)-th residual block and the input to the i-th residual
        block. Each tuple shoud be like: ``(n_filters, n_samples)``. `n_samples`
        for two consecutive samples should always decrease by an integer factor.
    dropout_rate: float [0, 1), optional
        Dropout rate used in all Dropout layers. Default is 0.8
    kernel_size: int, optional
        Kernel size for convolutional layers. The current implementation
        only supports odd kernel sizes. Default is 17.
    References
    ----------
    .. [1] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks,"
           arXiv:1603.05027, Mar. 2016. https://arxiv.org/pdf/1603.05027.pdf.
    .. [2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in 2016 IEEE Conference
           on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778. https://arxiv.org/pdf/1512.03385.pdf
    """

    def __init__(self, n_channels, n_samples, net_filter_size, net_seq_length, n_classes, kernel_size=17,
                 dropout_rate=0.5):
        super(ResNet1d, self).__init__()
        # my modifications!
        input_dim = (n_channels, n_samples)
        blocks_dim = list(zip(net_filter_size, net_seq_length))
        if n_classes == 2:
            n_classes = 1

        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                               stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

        # Linear layer
        n_filters_last, n_samples_last = blocks_dim[-1]
        last_layer_dim = n_filters_last * n_samples_last
        self.lin = nn.Linear(last_layer_dim, n_classes)
        self.n_blk = len(blocks_dim)

    def forward(self, x):

        x = x.squeeze(1)
        """Implement ResNet1d forward propagation"""
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)

        # Fully conected layer
        x = self.lin(x)
        return x


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
