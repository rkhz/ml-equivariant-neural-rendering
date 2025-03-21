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
        self.strides_2d = strides_2d
        self.channels_3d = channels_3d
        self.strides_3d = strides_3d
        self.num_channels_projection = num_channels_projection
        self.num_channels_inv_projection = num_channels_inv_projection
        self.mode = mode

        # Initialize layers

        # Inverse pass (image -> scene)
        # First transform image into a 2D representation
        self.inv_transform_2d = ResNet2d(self.img_shape, channels_2d,
                                         strides_2d)

        # Perform inverse projection from 2D to 3D
        input_shape = self.inv_transform_2d.output_shape
        self.inv_projection = InverseProjection(input_shape, num_channels_inv_projection)

        # Transform 3D inverse projection into a scene representation
        self.inv_transform_3d = ResNet3d(self.inv_projection.output_shape,
                                         channels_3d, strides_3d)
        # Add rotation layer
        self.rotation_layer = Rotate3d(self.mode)

        # Forward pass (scene -> image)
        # Forward renderer is just transpose of inverse renderer, so flip order
        # of channels and strides
        # Transform scene representation to 3D features
        forward_channels_3d = list(reversed(channels_3d))[1:] + [channels_3d[0]]
        forward_strides_3d = [-stride if abs(stride) == 2 else 1 for stride in list(reversed(strides_3d[1:]))] + [strides_3d[0]]
        self.transform_3d = ResNet3d(self.inv_transform_3d.output_shape,
                                     forward_channels_3d, forward_strides_3d)

        # Layer for projection of 3D representation to 2D representation
        self.projection = Projection(self.transform_3d.output_shape,
                                     num_channels_projection)

        # Transform 2D features to rendered image
        forward_channels_2d = list(reversed(channels_2d))[1:] + [channels_2d[0]]
        forward_strides_2d = [-stride if abs(stride) == 2 else 1 for stride in list(reversed(strides_2d[1:]))] + [strides_2d[0]]
        final_conv_channels_2d = img_shape[0]
        self.transform_2d = ResNet2d(self.projection.output_shape,
                                     forward_channels_2d, forward_strides_2d,
                                     final_conv_channels_2d)

        # Scene representation shape is output of inverse 3D transformation
        self.scene_shape = self.inv_transform_3d.output_shape
        # Add spherical mask before scene rotation
        self.spherical_mask = SphericalMask(self.scene_shape)

    def render(self, scene):
        """Renders a scene to an image.

        Args:
            scene (torch.Tensor): Shape (batch_size, channels, depth, height, width).
        """
        features_3d = self.transform_3d(scene)
        features_2d = self.projection(features_3d)
        return torch.sigmoid(self.transform_2d(features_2d))

    def inverse_render(self, img):
        """Maps an image to a (spherical) scene representation.

        Args:
            img (torch.Tensor): Shape (batch_size, channels, height, width).
        """
        # Transform image to 2D features
        features_2d = self.inv_transform_2d(img)
        # Perform inverse projection
        features_3d = self.inv_projection(features_2d)
        # Map 3D features to scene representation
        scene = self.inv_transform_3d(features_3d)
        # Ensure scene is spherical
        return self.spherical_mask(scene)


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
