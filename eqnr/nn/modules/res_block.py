import torch
import torch.nn as nn
from typing import Optional

__all__ = [
    "ResBlock2d",
    "ResBlock3d"
]


class _ResBlockNd(nn.Module):
    """Base Residual network block for N-dimensional convolutions.
    
    This module implements a residual connection where the input is added to the
    output of a residual block. Supports both pre-activation and post-activation
    residual blocks.
    
    Pre-activation residual blocks apply normalization and activation before convolutions,
    which can help with gradient flow in deeper networks and improve training stability.
    standard residual blocks apply these operations after convolutions, following the
    original ResNet design.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        pre_activation (bool, optional): Whether to use pre-activation residual blocks.
            Defaults to True.
        dim (int): Dimensionality of convolution (2 for 2D, 3 for 3D).
        device: Device to place the model on. 
            
    Note:
        Source: https://arxiv.org/abs/1603.05027

        When using pre_activation=True, activation is handled inside the residual block.
        
        When using pre_activation=False, activation is applied after the residual connection.
  
        When the input and output channels differ (in_channels != out_channels), a shortcut projection layer (using a 1×1 convolution without bias) is added to match the channel dimensions. In the standard (post-activation) design, normalization is applied to both the main path and the shortcut path after convolutions. By contrast, in the pre-activation design, normalization occurs only at the beginning of the block's main path and is intentionally omitted from the shortcut path. This design choice ensures uninterrupted gradient flow through the identity connection. The pre-activation approach improves gradient propagation in deep networks by keeping the shortcut path as clean as possible.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        pre_activation: bool=True, 
        dim: int=None,
        device=None
    ) -> None:
        super().__init__()
        assert dim in [2, 3], "dim must be 2 (Conv2d) or 3 (Conv3d)"
        
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pre_activation = pre_activation
        self.dim = dim
        
        # Configure residual mapping based on pre-activation 
        _ResMap = _PreActResMapNd if self.pre_activation else _ResMapNd
        self.residual_map = _ResMap(self.in_channels, self.out_channels, dim=self.dim, device=self.device)
        
        # 
        self._final_activation = None if self.pre_activation else nn.LeakyReLU(0.2, inplace=True)
        
        # Check if projection is needed (when input and output channels differ)
        if in_channels != out_channels:
            _Conv = {2: nn.Conv2d, 3: nn.Conv3d}[self.dim]
            self.shortcut = {
                True:   _Conv(in_channels, out_channels, kernel_size=1, stride=1, bias=False, device=self.device),
                False:  nn.Sequential(
                            _Conv(in_channels, out_channels, kernel_size=1, stride=1, bias=False, device=self.device),
                            nn.GroupNorm(num_groups=out_channels//8, num_channels=out_channels, device=self.device))
            }[self.pre_activation]
        else:
            self.shortcut = nn.Identity(device=self.device)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output of the residual block.
        """
        if self.pre_activation:
            return self.shortcut(x) + self.residual_map(x)
        else:
            return self._final_activation(self.shortcut(x) + self.residual_map(x))


class ResBlock2d(_ResBlockNd):
    """Residual network block with 2D convolutions.
    
    This module implements a residual connection where the input is added to the
    output of a residual block. Supports both pre-activation and post-activation
    residual blocks.
    
    Pre-activation residual blocks apply normalization and activation before convolutions,
    which can help with gradient flow in deeper networks and improve training stability.
    standard residual blocks apply these operations after convolutions, following the
    original ResNet design.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        pre_activation (bool, optional): Whether to use pre-activation residual blocks.
            Defaults to True.
        device: Device to place the model on. 

    Note:
        When using pre_activation=True, activation is handled inside the residual block.
        When using pre_activation=False, activation is applied after the residual connection.
    """
    def __init__(self, in_channels: int, out_channels: int, pre_activation: bool=True, device=None):
        super().__init__(in_channels, out_channels, pre_activation, dim=2, device=device)


class ResBlock3d(_ResBlockNd):
    """Residual network block with 3D convolutions.
    
    This module implements a residual connection where the input is added to the
    output of a residual block. Supports both pre-activation and post-activation
    residual blocks.
    
    Pre-activation blocks apply normalization and activation before convolutions,
    which can help with gradient flow in deeper networks and improve training stability.
    Post-activation blocks apply these operations after convolutions, following the
    original ResNet design.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        pre_activation (bool, optional): Whether to use pre-activation residual blocks.
            Defaults to True.
        device: Device to place the model on. 
            
    Note:
        When using pre_activation=True, activation is handled inside the residual block.
        When using pre_activation=False, activation is applied after the residual connection.
    """
    def __init__(self, in_channels: int, out_channels: int, pre_activation: bool=True, device=None):
        super().__init__(in_channels, out_channels, pre_activation, dim=3, device=device)


class _PreActResMapNd(nn.Module):
    """Base class for pre-activation residual function (residual mapping) with N-dimensional convolutions.
    
    Pre-activation means that normalization and activation functions are 
    applied before the convolutions rather than after them. This design can help 
    with gradient flow in deeper networks.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bottleneck_factor (float, optional): Scale factor for bottleneck channels. Default to 1.0
        dim (int): Dimensionality of convolution (2 for Conv2d, 3 for Conv3d).
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        bottleneck_factor: Optional[float]=1, 
        dim: int=None,
        device=None
    ) -> None:
        super(_PreActResMapNd, self).__init__()
        assert dim in [2, 3], "dim must be 2 (Conv2d) or 3 (Conv3d)"

        self.dim = dim
        self.device = device 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(bottleneck_factor * out_channels)
        
        _Conv = {2: nn.Conv2d, 3: nn.Conv3d}[self.dim]
        self.layers = nn.Sequential(
            nn.GroupNorm(num_groups=self.in_channels//8, num_channels=self.in_channels),
            nn.LeakyReLU(0.2, True),
            _Conv(self.in_channels, self.bottleneck_channels, 
                      kernel_size=1, stride=1, padding=0, bias=False),
            
            nn.GroupNorm(num_groups=self.bottleneck_channels//8, num_channels=self.bottleneck_channels),
            nn.LeakyReLU(0.2, True),
            _Conv(self.bottleneck_channels, self.bottleneck_channels, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            
            nn.GroupNorm(num_groups=self.bottleneck_channels//8, num_channels=self.bottleneck_channels),
            nn.LeakyReLU(0.2, True),
            _Conv(self.bottleneck_channels, self.out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=False)
        ).to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the pre-activation residual mapping.
        
        Args:
            inputs (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after processing through the residual mapping.
        """
        return self.layers(inputs)
            
 
class _ResMapNd(nn.Module):
    """Base class for standard residual function (residual mapping) with N-dimensional convolutions.
    
    This implementation follows the original ResNet design where normalization and activation
    are applied after convolutions.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bottleneck_factor (float, optional): Scale factor for bottleneck channels. Default to 1.0
        dim (int): Dimensionality of convolution (2 for Conv2d, 3 for Conv3d).
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        bottleneck_factor: Optional[float]=1, 
        dim: int=None,
        device=None
    ) -> None:
        super(_ResMapNd, self).__init__()
        assert dim in [2, 3], "dim must be 2 (Conv2d) or 3 (Conv3d)"
        
        self.dim = dim
        self.device = device 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(bottleneck_factor * out_channels) 
        
        _Conv = {2: nn.Conv2d, 3: nn.Conv3d}[self.dim]
        self.layers = nn.Sequential(
            _Conv(self.in_channels, self.bottleneck_channels, 
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=self.bottleneck_channels//8, num_channels=self.bottleneck_channels),
            nn.LeakyReLU(0.2, True),
            
            _Conv(self.bottleneck_channels, self.bottleneck_channels, 
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=self.bottleneck_channels//8, num_channels=self.bottleneck_channels),
            nn.LeakyReLU(0.2, True),
    
            _Conv(self.bottleneck_channels, self.out_channels, 
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(num_groups=self.out_channels//8, num_channels=self.out_channels)
        ).to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the standard residual mapping.
        
        Args:
            inputs (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after processing through the residual mapping.
        """
        return self.layers(inputs)
