import torch
import torch.nn.functional as F
from typing import Union


def rotate3d(
    volume: torch.Tensor,
    rotation_matrix: torch.Tensor, 
    mode: str = 'bilinear'
)-> torch.Tensor:
    """Performs 3D rotation of tensor volume by rotation matrix.

    Args:
        volume (torch.Tensor): Shape (batch_size, channels, depth, height, width).
        rotation_matrix (torch.Tensor): Batch of rotation matrices of shape
            (batch_size, 3, 3).
        mode (string): One of 'bilinear' and 'nearest' for interpolation mode
            used in grid_sample. Note that the 'bilinear' option actually
            performs trilinear interpolation.

    Notes:
        We use align_corners=False in grid_sample. See
        https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9
        for a nice illustration of why this is.
        
        The grid_sample function performs the inverse transformation of the input
        coordinates, so invert (ie.e transpose) matrix to get forward transformation:
            - (line 41): rotation_matrix.mT 
        
        The grid_sample function swaps x and z (i.e. it assumes the tensor
        dimensions are ordered as z, y, x), therefore we need to flip the rows and
        columns of the matrix (which we can verify is equivalent to multiplying by
        the appropriate permutation matrices):
            - (line 41): rotation_matrix.mT.flip(dims=(1,2))
            
        We use align_corners=False in affine_grid and grid_sample. For a nice illustration of why this is, see:
            https://discuss.pytorch.org/t/what-we-should-use-align-corners-false/22663/9
    """
    affine_matrix = F.pad(
        input=rotation_matrix.mT.flip(dims=(1,2)), 
        pad=(0, 1, 0, 0), value=0, mode='constant'
    )
    
    affine_grid = F.affine_grid(
        affine_matrix, 
        volume.shape,
        align_corners=False
    )
    
    return F.grid_sample(volume, affine_grid, mode=mode, align_corners=False)


def rotate_source_to_target(
    volume: torch.Tensor, 
    azimuth_source: torch.Tensor, elevation_source: torch.Tensor,
    azimuth_target: torch.Tensor, elevation_target: torch.Tensor, 
    mode: str = 'bilinear'
)-> torch.Tensor:
    """Performs 3D rotation matching two coordinate frames defined by a source
    view and a target view.

    Args:
        volume (torch.Tensor): Shape (batch_size, channels, depth, height, width).
        azimuth_source (torch.Tensor): Shape (batch_size,). Azimuth of source
            view in degrees.
        elevation_source (torch.Tensor): Shape (batch_size,). Elevation of
            source view in degrees.
        azimuth_target (torch.Tensor): Shape (batch_size,). Azimuth of target
            view in degrees.
        elevation_target (torch.Tensor): Shape (batch_size,). Elevation of
            target view in degrees.
    """
    rotation_matrix = _rotation_matrix_source_to_target(azimuth_source, elevation_source,
                                                        azimuth_target, elevation_target)
    return rotate3d(volume, rotation_matrix, mode=mode)


def _rotation_matrix_source_to_target(
    azimuth_source: torch.Tensor, elevation_source: torch.Tensor,
    azimuth_target: torch.Tensor, elevation_target: torch.Tensor
) -> torch.Tensor:
    """
    Returns rotation matrix matching two views defined by azimuth, elevation
    pairs.

    Args:
        azimuth_source (torch.Tensor): Shape (batch_size,). Azimuth of source
            view in degrees.
        elevation_source (torch.Tensor): Shape (batch_size,). Elevation of
            source view in degrees.
        azimuth_target (torch.Tensor): Shape (batch_size,). Azimuth of target
            view in degrees.
        elevation_target (torch.Tensor): Shape (batch_size,). Elevation of
            target view in degrees.
    Notes:
        Calculate rotation matrix bringing source view to target view (note that
        for rotation matrix, inverse is transpose)
    """
    rotation_source = _rotation_matrix_camera_to_world(azimuth_source, elevation_source)
    rotation_target = _rotation_matrix_camera_to_world(azimuth_target, elevation_target)
    return rotation_target @ (rotation_source.mT)


def _rotation_matrix_camera_to_world(
    azimuth: torch.Tensor, 
    elevation: torch.Tensor
) -> torch.Tensor:
    """
    Returns rotation matrix matching the default view (i.e. both azimuth and
    elevation are zero) to the view defined by the azimuth, elevation pair.


    Args:
        azimuth (torch.Tensor): Shape (batch_size,). Azimuth of camera in
            degrees.
        elevation (torch.Tensor): Shape (batch_size,). Elevation of camera in
            degrees.

    Notes:
        The azimuth and elevation refer to the position of the camera. This
        function returns the rotation of the *scene representation*, i.e. the
        inverse of the camera transformation.

        In the coordinate system defined, azimuth rotation corresponds to negative rotation 
        about y axis and elevation rotation to a negative rotation about z axis.
        
        We first perform elevation rotation followed by azimuth when rotating camera.
        Object rotation matrix is inverse (i.e. transpose) of the camera rotation matrix:
            - rotation_matrix_camera = _rotation_matrix_y(-azimuth) @ _rotation_matrix_z(-elevation)
            - rotation_matrix_world (object) = rotation_matrix_camera.mT 
    """
    return (_rotation_matrix_y(-azimuth) @ _rotation_matrix_z(-elevation)).mT


def _rotation_matrix_y(
    angle_degrees: Union[torch.Tensor, float]
) -> torch.Tensor:
    """
    Create a 3D rotation matrix for rotation around the Y-axis.
    
    Args:
        angle_degrees (torch.Tensor or float): Angle(s) in degrees of shape (batch_size,) or (1,).
        
    Returns:
        torch.Tensor: A rotation matrix of shape (batch_size, 3, 3) or (1, 3, 3).
    """
    angle = torch.deg2rad(angle_degrees)
    
    cos_ = torch.cos(angle)
    sin_ = torch.sin(angle)
    
    rotation_matrix = torch.zeros((*angle.shape, 3, 3), dtype=torch.float32, device=angle.device)

    rotation_matrix[:, 0, 0] =  cos_
    rotation_matrix[:, 0, 2] =  sin_
    rotation_matrix[:, 1, 1] =  1.0
    rotation_matrix[:, 2, 0] = -sin_
    rotation_matrix[:, 2, 2] =  cos_
    return rotation_matrix


def _rotation_matrix_z(
    angle_degrees: Union[torch.Tensor, float]
) -> torch.Tensor:
    """
    Create a 3D rotation matrix for rotation around the Z-axis.
    
    Args:
        angle_degrees (torch.Tensor or float): Angle(s) in degrees of shape (batch_size,) or (1,).
        
    Returns:
        torch.Tensor: A rotation matrix of shape (batch_size, 3, 3) or (1, 3, 3).
    """
    angle = torch.deg2rad(angle_degrees)
    
    cos_ = torch.cos(angle)
    sin_ = torch.sin(angle)
    
    rotation_matrix = torch.zeros((*angle.shape, 3, 3), dtype=torch.float32, device=angle.device)
    rotation_matrix[:, 0, 0] =  cos_
    rotation_matrix[:, 0, 1] = -sin_
    rotation_matrix[:, 1, 0] =  sin_
    rotation_matrix[:, 1, 1] =  cos_
    rotation_matrix[:, 2, 2] =  1.0
    return rotation_matrix


def _get_num_groups(
    num_channels: int
    ) -> int:
    """Returns number of groups to use in a GroupNorm layer with a given number
    of channels. Note that these choices are hyperparameters.

    Args:
        num_channels (int): Number of channels.
    """

    thresholds = [8, 32, 64, 128, 256]
    num_groups = [1, 2, 4, 8, 16, 32]

    return num_groups[sum(num_channels >= t for t in thresholds)]
