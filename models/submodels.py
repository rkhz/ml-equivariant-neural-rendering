import torch
import torch.nn as nn
from eqnr.nn.functional import _get_num_groups
from eqnr.nn import ResBlock2d, ResBlock3d


class ResNet2d(nn.Module):
    """ResNets for 2d inputs.

    Args:
        input_shape (tuple of ints): Shape of the input to the model. Should be
            of the form (channels, height, width).
        channels (tuple of ints): List of number of channels for each layer.
            Length of this tuple corresponds to number of layers in network.
        strides (tuple of ints): List of strides for each layer. Length of this
            tuple corresponds to number of layers in network. If stride is 1, a
            residual layer is applied. If stride is 2 a convolution with stride
            2 is applied. If stride is -2 a transpose convolution with stride 2
            is applied.
        final_conv_channels (int): If not 0, a convolution is added as the final
            layer, with the number of output channels specified by this int.
        filter_multipliers (tuple of ints): Multipliers for filters in residual
            layers.
        add_groupnorm (bool): If True, adds GroupNorm layers.


    Notes:
        The first layer of this model is a standard convolution to increase the
        number of filters. A convolution can optionally be added at the final
        layer.
    """
    def __init__(self, input_shape, channels, strides, final_conv_channels=0,
                 filter_multipliers=(1, 1), add_groupnorm=True):
        super(ResNet2d, self).__init__()
        assert len(channels) == len(strides), "Length of channels tuple is {} and length of strides tuple is {} but " \
                                              "they should be equal".format(len(channels), len(strides))
        self.input_shape = input_shape
        self.channels = channels
        self.strides = strides
        self.filter_multipliers = filter_multipliers
        self.add_groupnorm = add_groupnorm

        # Calculate output_shape:
        # Every layer with stride 2 divides the height and width by 2.
        # Similarly, every layer with stride -2 multiplies the height and width
        # by 2
        output_channels, output_height, output_width = input_shape

        for stride in strides:
            if stride == 1:
                pass
            elif stride == 2:
                output_height //= 2
                output_width //= 2
            elif stride == -2:
                output_height *= 2
                output_width *= 2

        self.output_shape = (channels[-1], output_height, output_width)

        # Build layers
        # First layer to increase number of channels before applying residual
        # layers
        forward_layers = [
            nn.Conv2d(self.input_shape[0], channels[0], kernel_size=1,
                      stride=1, padding=0)
        ]
        in_channels = channels[0]
        multiplier1x1, multiplier3x3 = filter_multipliers
        for out_channels, stride in zip(channels, strides):
            if stride == 1:
                forward_layers.append(
                    ResBlock2d(in_channels,
                              [out_channels * multiplier1x1, out_channels * multiplier3x3],
                               group_norm=add_groupnorm)
                )
            if stride == 2:
                forward_layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=4,
                              stride=2, padding=1)
                )
            if stride == -2:
                forward_layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                              stride=2, padding=1)
                )

            # Add non-linearity
            if stride == 2 or stride == -2:
                forward_layers.append(nn.GroupNorm(_get_num_groups(out_channels), out_channels))
                forward_layers.append(nn.LeakyReLU(0.2, True))

            in_channels = out_channels

        if final_conv_channels:
            forward_layers.append(
                nn.Conv2d(in_channels, final_conv_channels, kernel_size=1,
                          stride=1, padding=0)
            )

        self.forward_layers = nn.Sequential(*forward_layers)

    def forward(self, inputs):
        """Applies ResNet to image-like features.

        Args:
            inputs (torch.Tensor): Image-like tensor, with shape (batch_size,
                channels, height, width).
        """
        return self.forward_layers(inputs)


class ResNet3d(nn.Module):
    """ResNets for 3d inputs.

    Args:
        input_shape (tuple of ints): Shape of the input to the model. Should be
            of the form (channels, depth, height, width).
        channels (tuple of ints): List of number of channels for each layer.
            Length of this tuple corresponds to number of layers in network.
            Note that this corresponds to number of *output* channels for each
            convolutional layer.
        strides (tuple of ints): List of strides for each layer. Length of this
            tuple corresponds to number of layers in network. If stride is 1, a
            residual layer is applied. If stride is 2 a convolution with stride
            2 is applied. If stride is -2 a transpose convolution with stride 2
            is applied.
        final_conv_channels (int): If not 0, a convolution is added as the final
            layer, with the number of output channels specified by this int.
        filter_multipliers (tuple of ints): Multipliers for filters in residual
            layers.
        add_groupnorm (bool): If True, adds GroupNorm layers.

    Notes:
        The first layer of this model is a standard convolution to increase the
        number of filters. A convolution can optionally be added at the final
        layer.
    """
    def __init__(self, input_shape, channels, strides, final_conv_channels=0,
                 filter_multipliers=(1, 1), add_groupnorm=True):
        super(ResNet3d, self).__init__()
        assert len(channels) ==  len(strides), "Length of channels tuple is {} and length of strides tuple is {} but they should be equal".format(len(channels), len(strides))
        self.input_shape = input_shape
        self.channels = channels
        self.strides = strides
        self.filter_multipliers = filter_multipliers
        self.add_groupnorm = add_groupnorm

        # Calculate output_shape
        output_channels, output_depth, output_height, output_width = input_shape

        for stride in strides:
            if stride == 1:
                pass
            elif stride == 2:
                output_depth //= 2
                output_height //= 2
                output_width //= 2
            elif stride == -2:
                output_depth *= 2
                output_height *= 2
                output_width *= 2

        self.output_shape = (channels[-1], output_depth, output_height, output_width)

        # Build layers
        # First layer to increase number of channels before applying residual
        # layers
        forward_layers = [
            nn.Conv3d(self.input_shape[0], channels[0], kernel_size=1,
                      stride=1, padding=0)
        ]
        in_channels = channels[0]
        multiplier1x1, multiplier3x3 = filter_multipliers
        for out_channels, stride in zip(channels, strides):
            if stride == 1:
                forward_layers.append(
                    ResBlock3d(in_channels,
                              [out_channels * multiplier1x1, out_channels * multiplier3x3],
                               group_norm=add_groupnorm)
                )
            if stride == 2:
                forward_layers.append(
                    nn.Conv3d(in_channels, out_channels, kernel_size=4,
                              stride=2, padding=1)
                )
            if stride == -2:
                forward_layers.append(
                    nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4,
                              stride=2, padding=1)
                )

            # Add non-linearity
            if stride == 2 or stride == -2:
                forward_layers.append(nn.GroupNorm(_get_num_groups(out_channels), out_channels))
                forward_layers.append(nn.LeakyReLU(0.2, True))

            in_channels = out_channels

        if final_conv_channels:
            forward_layers.append(
                nn.Conv3d(in_channels, final_conv_channels, kernel_size=1,
                          stride=1, padding=0)
            )

        self.forward_layers = nn.Sequential(*forward_layers)

    def forward(self, inputs):
        """Applies ResNet to 3D features.

        Args:
            inputs (torch.Tensor): Tensor, with shape (batch_size, channels,
                depth, height, width).
        """
        return self.forward_layers(inputs)


