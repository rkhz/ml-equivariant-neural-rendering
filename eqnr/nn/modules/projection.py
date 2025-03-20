import torch.nn as nn
from eqnr.nn.functional import _get_num_groups

__all__ = [
    "Projection",
    "InverseProjection"
]

class Projection(nn.Module):
    """Performs a projection from a 3D voxel-like feature map to a 2D image-like
    feature map.

    Args:
        input_shape (tuple of ints): Shape of 3D input, (channels, depth,
            height, width).
        num_channels (tuple of ints): Number of channels in each layer of the
            projection unit.

    Notes:
        This layer is inspired by the Projection Unit from
        https://arxiv.org/abs/1806.06575.
    """
    def __init__(self, input_shape, num_channels):
        super(Projection, self).__init__()
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.output_shape = (num_channels[-1],) + input_shape[2:]
        # Number of input channels for first 2D convolution is
        # channels * depth since we flatten the 3D input
        in_channels = self.input_shape[0] * self.input_shape[1]
        # Initialize forward pass layers
        forward_layers = []
        num_layers = len(num_channels)
        for i in range(num_layers):
            out_channels = num_channels[i]
            forward_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
            # Add non linearites, except for last layer
            if i != num_layers - 1:
                forward_layers.append(nn.GroupNorm(_get_num_groups(out_channels), out_channels))
                forward_layers.append(nn.LeakyReLU(0.2, True))
            in_channels = out_channels
        # Set up forward layers as model
        self.forward_layers = nn.Sequential(*forward_layers)

    def forward(self, inputs):
        """Reshapes inputs from 3D -> 2D and applies 1x1 convolutions.

        Args:
            inputs (torch.Tensor): Voxel like tensor, with shape (batch_size,
                channels, depth, height, width).
        """
        batch_size, channels, depth, height, width = inputs.shape
        # Reshape 3D -> 2D
        reshaped = inputs.view(batch_size, channels * depth, height, width)
        # 1x1 conv layers
        return self.forward_layers(reshaped)


class InverseProjection(nn.Module):
    """Performs an inverse projection from a 2D feature map to a 3D feature map.

    Args:
        input_shape (tuple of ints): Shape of 2D input, (channels, height, width).
        num_channels (tuple of ints): Number of channels in each layer of the
            projection unit.

    Note:
        The depth will be equal to the height and width of the input map.
        Therefore, the final number of channels must be divisible by the height
        and width of the input.
    """
    def __init__(self, input_shape, num_channels):
        super(InverseProjection, self).__init__()
        self.input_shape = input_shape
        self.num_channels = num_channels
        assert num_channels[-1] % input_shape[-1] == 0, "Number of output channels is {} which is not divisible by " \
                                                        "width {} of image".format(num_channels[-1], input_shape[-1])
        self.output_shape = (num_channels[-1] // input_shape[-1], input_shape[-1]) + input_shape[1:]

        # Initialize forward pass layers
        in_channels = self.input_shape[0]
        forward_layers = []
        num_layers = len(num_channels)
        for i in range(num_layers):
            out_channels = num_channels[i]
            forward_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                          padding=0)
            )
            # Add non linearites, except for last layer
            if i != num_layers - 1:
                forward_layers.append(nn.GroupNorm(_get_num_groups(out_channels), out_channels))
                forward_layers.append(nn.LeakyReLU(0.2, True))
            in_channels = out_channels
        # Set up forward layers as model
        self.forward_layers = nn.Sequential(*forward_layers)

    def forward(self, inputs):
        """Applies convolutions and reshapes outputs from 2D -> 3D.

        Args:
            inputs (torch.Tensor): Image like tensor, with shape (batch_size,
                channels, height, width).
        """
        # 1x1 conv layers
        features = self.forward_layers(inputs)
        # Reshape 3D -> 2D
        batch_size = inputs.shape[0]
        return features.view(batch_size, *self.output_shape)