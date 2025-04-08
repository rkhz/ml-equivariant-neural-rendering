import torch.nn as nn
import einops.layers.torch as einops_nn

from .conv_block import ConvBlock2d, ConvBlock3d

__all__ = [
    "Project2DTo3D",
    "Project3DTo2D"
]


class Project2DTo3D(nn.Module):
    """
    Notes:
        This layer is inspired by the Projection Unit from
        https://arxiv.org/abs/1806.06575.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        pre_activation: bool=True, 
        device=None
    ) -> None:
        super(Project2DTo3D, self).__init__()
        super().__init__()
        
        self.device = device
        self.in_channels= in_channels 
        self.out_channels = out_channels  #128
        
        self.layers = nn.Sequential(
            ConvBlock2d(self.in_channels, 256, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),
            ConvBlock2d(256, 512, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),
            ConvBlock2d(512,1024, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),
            einops_nn.Rearrange('b (c d) h w -> b c d h w', c=self.out_channels),
        )

    def forward(self, x):
        return self.layers(x)



class Project3DTo2D(nn.Module):
    """
    Notes:
        This layer is inspired by the Projection Unit from
        https://arxiv.org/abs/1806.06575.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        pre_activation: bool=True, 
        device=None
    ) -> None:
        super(Project3DTo2D, self).__init__()
        super().__init__()
        
        self.device = device
        self.in_channels= in_channels 
        self.out_channels = out_channels  #128
        
        self.layers = nn.Sequential(
            einops_nn.Rearrange('b c d h w -> b (c d) h w', c=self.in_channels),
            ConvBlock2d(1024, 512, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),   # same as doing Conv actually 
            ConvBlock2d( 512, 256, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation),   # same as doing Conv actually 
            ConvBlock2d( 256, 128, kernel_size=1, stride=1, padding=0, pre_activation=pre_activation)    # same as doing Conv actually 
        )

    def forward(self, x):
        return self.layers(x)

    
# class _Reshape(nn.Module):
#     def __init__(
#         self, 
#         final_channels: int,
#         to_dim: int,
#     ) -> None:
#         super().__init__()
        
#         self.final_channels = final_channels 
#         self.to_dim = to_dim
    
#     def forward(self, x):
#         if self.to_dim == 3:
#             return x.view(-1, self.final_channels, x.shape[1] // self.final_channels, x.shape[2], x.shape[3])
#         else: # self.to_dim == 2
#             #assert self.final_channels == x.shape[1] * x.shape[2]
#             return x.view(-1, self.final_channels * x.shape[2], x.shape[3], x.shape[4])
    