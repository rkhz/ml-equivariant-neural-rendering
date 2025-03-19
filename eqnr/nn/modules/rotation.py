import torch.nn as nn
from eqnr.nn.functional import rotate3d, rotate_source_to_target


class Rotate3d(nn.Module):
    """Layer used to rotate 3D feature maps.

    Args:
        mode (string): One of 'bilinear' and 'nearest' for interpolation mode
            used when resampling rotated values on the grid.
    """
    def __init__(self, mode='bilinear'):
        super(Rotate3d, self).__init__()
        self.mode = mode

    def forward(self, volume, rotation_matrix):
        """Rotates the volume by the rotation matrix.

        Args:
            volume (torch.Tensor): Shape (batch_size, channels, depth, height, width).
            rotation_matrix (torch.Tensor): Batch of rotation matrices of shape
                (batch_size, 3, 3).
        """
        return rotate3d(volume, rotation_matrix, mode=self.mode)

    def rotate_source_to_target(self, volume, azimuth_source, elevation_source,
                                azimuth_target, elevation_target):
        """Rotates volume from source coordinate frame to target coordinate
        frame.

        Args:
            volume (torch.Tensor): Shape (batch_size, channels, depth, height, width).
            azimuth_source (torch.Tensor): Shape (batch_size,). Azimuth of
                source view in degrees.
            elevation_source (torch.Tensor): Shape (batch_size,). Elevation of
                source view in degrees.
            azimuth_target (torch.Tensor): Shape (batch_size,). Azimuth of
                target view in degrees.
            elevation_target (torch.Tensor): Shape (batch_size,). Elevation of
                target view in degrees.
        """
        return rotate_source_to_target(volume, azimuth_source, elevation_source,
                                       azimuth_target, elevation_target,
                                       mode=self.mode)


