import torch
from typing import Union

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
            - rotation_matrix_world (object) = rotation_matrix_camera.transpose(1, 2) 
    """
    return (_rotation_matrix_y(-azimuth) @ _rotation_matrix_z(-elevation)).transpose(1, 2)



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
    return rotation_target @ (rotation_source.transpose(1, 2))