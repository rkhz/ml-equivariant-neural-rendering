import torch
import torch.nn as nn

from eqnr.nn import SphericalMask, Rotate3d, InverseProjection, Projection
from models.submodels import ResNet2d, ResNet3d


class NeuralRenderer(nn.Module):
    """Implements a Neural Renderer with an implicit scene representation that
    allows both forward and inverse rendering.

    The forward pass from 3d scene to 2d image is (rendering):
    Scene representation (input) -> ResNet3d -> Projection -> ResNet2d ->
    Rendered image (output)

    The inverse pass from 2d image to 3d scene is (inverse rendering):
    Image (input) -> ResNet2d -> Inverse Projection -> ResNet3d -> Scene
    representation (output)

    Args:
        img_shape (tuple of ints): Shape of the image input to the model. Should
            be of the form (channels, height, width).
        channels_2d (tuple of ints): List of channels for 2D layers in inverse
            rendering model (image -> scene).
        strides_2d (tuple of ints): List of strides for 2D layers in inverse
            rendering model (image -> scene).
        channels_3d (tuple of ints): List of channels for 3D layers in inverse
            rendering model (image -> scene).
        strides_3d (tuple of ints): List of channels for 3D layers in inverse
            rendering model (image -> scene).
        num_channels_inv_projection (tuple of ints): Number of channels in each
            layer of inverse projection unit from 2D to 3D.
        num_channels_projection (tuple of ints): Number of channels in each
            layer of projection unit from 2D to 3D.
        mode (string): One of 'bilinear' and 'nearest' for interpolation mode
            used when rotating voxel grid.

    Notes:
        Given the inverse rendering channels and strides, the model will
        automatically build a forward renderer as the transpose of the inverse
        renderer.
    """
    def __init__(self, img_shape, channels_2d, strides_2d, channels_3d,
                 strides_3d, num_channels_inv_projection, num_channels_projection,
                 mode='bilinear'):
        super(NeuralRenderer, self).__init__()
        self.img_shape = img_shape
        
        self.channels_2d = channels_2d
        self.channels_3d = channels_3d
        
        self.strides_2d = strides_2d
        self.strides_3d = strides_3d
        
        self.num_channels_projection = num_channels_projection
        self.num_channels_inv_projection = num_channels_inv_projection
        
        self.mode = mode
        
        # Initialize layers

        # 1. Inverse pass (image -> scene) ---#
        #--- First transform image into a 2D representation
        self.inv_transform_2d = ResNet2d(self.img_shape, channels_2d, strides_2d)

        #---  Perform inverse projection from 2D to 3D
        self.inv_projection = InverseProjection(self.inv_transform_2d.output_shape, num_channels_inv_projection)

        #---  Transform 3D inverse projection into a scene representation
        self.inv_transform_3d = ResNet3d(self.inv_projection.output_shape, channels_3d, strides_3d)
        
        #--- Scene representation shape is output of inverse 3D transformation
        self.scene_shape = self.inv_transform_3d.output_shape
        self.spherical_mask = SphericalMask(self.scene_shape)
        
        self.inverse_render = nn.Sequential(
            self.inv_transform_2d,  # get features_2d: ransform image to 2D features
            self.inv_projection,    # get features_3d: erform inverse projection
            self.inv_transform_3d,  # get scene: map 3D features to scene representation
            self.spherical_mask     # Ensure scene is spherical
        )
        
        # 2. Add rotation layer
        self.rotation_layer = Rotate3d(self.mode)

        # 3. Forward pass (scene -> image)
        #---  Forward renderer is just transpose of inverse renderer, so flip order of channels and strides
        #--- Transform scene representation to 3D features
        forward_channels_3d = list(reversed(channels_3d))[1:] + [channels_3d[0]]
        forward_strides_3d = [-stride if abs(stride) == 2 else 1 for stride in list(reversed(strides_3d[1:]))] + [strides_3d[0]]
        self.transform_3d = ResNet3d(self.scene_shape, forward_channels_3d, forward_strides_3d)

        #--- Layer for projection of 3D representation to 2D representation
        self.projection = Projection(self.transform_3d.output_shape, num_channels_projection)

        #--- Transform 2D features to rendered image
        forward_channels_2d = list(reversed(channels_2d))[1:] + [channels_2d[0]]
        forward_strides_2d = [-stride if abs(stride) == 2 else 1 for stride in list(reversed(strides_2d[1:]))] + [strides_2d[0]]
        final_conv_channels_2d = img_shape[0]
        self.transform_2d = ResNet2d(self.projection.output_shape, forward_channels_2d, forward_strides_2d, final_conv_channels_2d)


        self.render = torch.nn.Sequential(
            self.transform_3d,
            self.projection,  
            self.transform_2d, 
            torch.nn.Sigmoid()
        )



    def forward(self, imgs, **kwargs):
        scenes = self.inverse_render(imgs)
        scenes_rotated = self.rotation_layer(scenes, **kwargs)
        rendered = self.render(scenes_rotated)
        return imgs, rendered, scenes, scenes_rotated
    
 
    def get_model_config(self):
        """Returns the complete model configuration as a dict."""
        return {
            "img_shape": self.img_shape,
            "channels_2d": self.channels_2d,
            "strides_2d": self.strides_2d,
            "channels_3d": self.channels_3d,
            "strides_3d": self.strides_3d,
            "num_channels_inv_projection": self.num_channels_inv_projection,
            "num_channels_projection": self.num_channels_projection,
            "mode": self.mode
        }

    def save(self, filename):
        """Saves model and its config.

        Args:
            filename (string): Path where model will be saved. Should end with
                '.pt' or '.pth'.
        """
        torch.save({
            "config": self.get_model_config(),
            "state_dict": self.state_dict()
        }, filename)
