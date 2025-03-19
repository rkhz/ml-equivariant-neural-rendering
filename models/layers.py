import torch.nn as nn


class _ConvBlock(nn.Module):
    """Base class for ConvBlock2d and ConvBlock3d.
    
    Args:
        in_channels (int): Number of input channels.
        num_filters (list of ints): Number of filters for each conv layer.
        group_norm (bool): Whether to apply GroupNorm.
        dim (int): 2 for Conv2d, 3 for Conv3d.
    """
    def __init__(self, in_channels, num_filters, group_norm=True, dim=None):
        super(_ConvBlock, self).__init__()
        assert dim in [2, 3], "dim must be 2 (Conv2d) or 3 (Conv3d)"
        
        _kernel_size = [1, 3, 1]
        _padding = [0, 1, 0]
        _bias = False if group_norm else True

        # Select correct convolution layer (Conv2d or Conv3d)
        conv_layer = nn.Conv2d if dim == 2 else nn.Conv3d

        layers = []
        prev_in_channels = in_channels
        for i, out_channels in enumerate(num_filters + [in_channels]):
            if group_norm:
                num_groups = _get_num_groups(prev_in_channels)
                layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=prev_in_channels))

            layers.append(nn.LeakyReLU(0.2, True))
            layers.append(conv_layer(prev_in_channels, out_channels, kernel_size=_kernel_size[i], padding=_padding[i], stride=1, bias=_bias))
            
            prev_in_channels = out_channels

        self.forward_layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.forward_layers(inputs)

class ConvBlock2d(_ConvBlock):
    def __init__(self, in_channels, num_filters, group_norm=True):
        super().__init__(in_channels, num_filters, group_norm, dim=2)


class ConvBlock3d(_ConvBlock):
    def __init__(self, in_channels, num_filters, group_norm=True):
        super().__init__(in_channels, num_filters, group_norm, dim=3)
            

class ResBlock2d(nn.Module):
    """Residual block of 1x1, 3x3, 1x1 convolutions with non linearities. Shape
    of input and output is the same.

    Args:
        in_channels (int): Number of channels in input.
        num_filters (list of ints): List of two ints with the number of filters
            for the first and second conv layers. Third conv layer must have the
            same number of input filters as there are channels.
        add_groupnorm (bool): If True adds GroupNorm.
    """
    def __init__(self, in_channels, num_filters, group_norm=True):
        super(ResBlock2d, self).__init__()
        self.residual_layers = ConvBlock2d(in_channels, num_filters, group_norm)

    def forward(self, inputs):
        return inputs + self.residual_layers(inputs)

class ResBlock3d(nn.Module):
    """Residual block of 1x1, 3x3, 1x1 convolutions with non linearities. Shape
    of input and output is the same.

    Args:
        in_channels (int): Number of channels in input.
        num_filters (list of ints): List of two ints with the number of filters
            for the first and second conv layers. Third conv layer must have the
            same number of input filters as there are channels.
        add_groupnorm (bool): If True adds GroupNorm.
    """
    def __init__(self, in_channels, num_filters, group_norm=True):
        super(ResBlock3d, self).__init__()
        self.residual_layers = ConvBlock3d(in_channels, num_filters, group_norm)

    def forward(self, inputs):
        return inputs + self.residual_layers(inputs)


    
def _get_num_groups(num_channels):
    """Returns number of groups to use in a GroupNorm layer with a given number
    of channels. Note that these choices are hyperparameters.

    Args:
        num_channels (int): Number of channels.
    """

    thresholds = [8, 32, 64, 128, 256]
    num_groups = [1, 2, 4, 8, 16, 32]
    
    return num_groups[sum(num_channels >= t for t in thresholds)]