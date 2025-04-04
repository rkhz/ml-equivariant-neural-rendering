import torch
import torch.nn as nn

__all__ = [
    "SphericalMask"
]


class SphericalMask(nn.Module):
    """Sets all features outside the largest sphere embedded in a cubic tensor
    to zero.

    Args:
        input_shape (tuple of ints): Shape of 3D feature map. Should have the
            form (channels, depth, height, width).
        radius_fraction (float): Fraction of radius to keep as non zero. E.g.
            if radius_fraction=0.9, only elements within the sphere of radius
            0.9 of half the cube length will not be zeroed. Must be in [0., 1.].
    """
    def __init__(
        self, 
        in_channels, 
        latent_dim,
        radius_fraction=1.0,
        device=None
    ) -> None:
        super(SphericalMask, self).__init__()
        
        self.device = device
        self.in_channels = in_channels
        self.latent_dim = latent_dim 
        
        self.input_shape = (self.in_channels, self.latent_dim, self.latent_dim, self.latent_dim)

        depth = height = width = self.latent_dim
        # Build spherical mask
        mask = torch.ones(self.input_shape, device=self.device)
        mask_center = (depth - 1) / 2  # Center of cube
        radius = (depth - 1) / 2  # Sphere radius

        #-- calculate squared radius fraction once
        radius_squared = (radius_fraction * radius) ** 2

        #-- use torch operations to vectorize the loop
        indices = torch.meshgrid(torch.arange(depth), torch.arange(height), torch.arange(width), indexing="ij")
        squared_distances = (indices[0] - mask_center) ** 2 + (indices[1] - mask_center) ** 2 + (indices[2] - mask_center) ** 2

        #- apply mask condition
        mask[:, squared_distances > radius_squared] = 0

        # Register buffer adds a key to the state dict of the model. This will
        # track the attribute without registering it as a learnable parameter.
        # This also means mask will be moved to device when calling
        # model.to(device)
        self.register_buffer('mask', mask)

    def forward(self, volume):
        """Applies a spherical mask to input.

        Args:
            volume (torch.Tensor): Shape (batch_size, channels, depth, height, width).
        """
        return volume * self.mask

