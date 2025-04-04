import torch
import eqnr.nn.functional as eqnr_F

class Rotate3d(torch.nn.Module):
    """
    Layer used to rotate 3D feature maps.
    
    This module provides functionality to rotate 3D volumes (feature maps) using either
    direct rotation matrices or source-to-target angle pairs. It supports different
    behaviors for training and evaluation modes.
    
    In training mode:
        It implements a pairwise swapping mechanism designed for batches
        where consecutive pairs of examples represent the same scene from different viewpoints.
        For example, in a batch:
            - Items 0,1: Same scene A from two different viewpoints
            - Items 2,3: Same scene B from two different viewpoints
            - And so on...
    
        The swapping mechanism rotates each scene's representation to match its pair's viewpoint,
        which enables equivariance constraints during training.
    
    In evaluation mode:
        It supports either direct rotation matrix application or rotation 
        based on source and target viewing angles.
    
    Args:
        mode (str): Interpolation mode used when resampling rotated values on the grid.
                   Must be one of 'bilinear' or 'nearest'. Default: 'bilinear'
    
    Attributes:
        mode (str): The interpolation mode.
        _pairwise_swap_indices (list): Indices used for pairwise swapping in a batch during training.
                                       Initialized as None and set during the first forward pass.
                                       For a batch size of N, this creates pairs: [1,0,3,2,...,N-1,N-2]
    
    """
    def __init__(self, mode: str='bilinear'):
        super(Rotate3d, self).__init__()
        self.mode = mode
        self._pairwise_swap_indices = None

        
    def forward(self, volume: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Unified forward pass that routes to the appropriate implementation.
        
        This method determines whether to use training or evaluation specific logic
        based on the module's training state, then routes to the appropriate
        implementation.
        
        Args:
            volume (torch.Tensor): The 3D volume to rotate. Shape: (batch_size, channels, depth, height, width)
            **kwargs (additional arguments that depend on the mode):
                In training mode:
                    azimuth (torch.Tensor): Azimuth angles in radians. Shape: (batch_size,)
                    elevation (torch.Tensor): Elevation angles in radians. Shape: (batch_size,)
                
                In evaluation mode (one of the following sets is required):
                    rotation_matrix (torch.Tensor): Batch of rotation matrices. Shape: (batch_size, 3, 3)
                    OR
                    azimuth_source (torch.Tensor): Source azimuth angles. Shape: (batch_size,)
                    elevation_source (torch.Tensor): Source elevation angles. Shape: (batch_size,)
                    azimuth_target (torch.Tensor): Target azimuth angles. Shape: (num_views,)
                    elevation_target (torch.Tensor): Target elevation angles. Shape: (num_views,)
        
        Returns:
            torch.Tensor: The rotated volume with the same shape as input,
                          or expanded to (num_views, channels, depth, height, width) 
                          if generating multiple views.
                         
        Note:
            For training mode, the batch is expected to contain consecutive pairs of the 
            same scene from different viewpoints (e.g., scene A at angle 1, scene A at angle 2,
            scene B at angle 1, scene B at angle 2, etc.)
            
            When generating new views in evaluation mode (azimuth_source.shape[0] != azimuth_target.shape[0]),
            the input volume is expanded to match the number of target views.
        """
        if self.training:
            return self._train_forward(volume, kwargs['azimuth'], kwargs['elevation'])
        else:
            return self._eval_forward(volume, **kwargs)
        
        
    def _train_forward(self, volume: torch.Tensor, azimuth: torch.Tensor, elevation: torch.Tensor) -> torch.Tensor:
        """
        Training-specific forward pass using pairwise viewpoint swapping.
        
        In training mode, this method:
            1. Creates rotation matrices to rotate each scene representation to its paired
            example's viewpoint (using the swap mechanism)
            2. Applies the rotation to transform each scene to the viewpoint of its pair
            3. Swaps the results back to align with the original batch order
        
        For example, with scenes z1, z2 (same scene, different viewpoints):
            - We rotate z1 to z1' (z1 from viewpoint of z2)
            - We rotate z2 to z2' (z2 from viewpoint of z1)
            - We swap to get [z2', z1'] which should closely match [z1, z2]
        
        This enables equivariance constraints during training.
        
        Args:
            volume (torch.Tensor): The 3D volume to rotate. Shape: (batch_size, channels, depth, height, width)
            azimuth (torch.Tensor): Azimuth angles in radians. Shape: (batch_size,)
            elevation (torch.Tensor): Elevation angles in radians. Shape: (batch_size,)
            
        Returns:
            torch.Tensor: The rotated and swapped volume with the same shape as input.
        """
        if self._pairwise_swap_indices is None or len(self._pairwise_swap_indices) != volume.shape[0]:
                self._pairwise_swap_indices = [idx^1 for idx in range(volume.shape[0])]
                
        # Create rotation matrices that transform each scene to its pair's viewpoint
        rotation_matrix = eqnr_F.rotation_matrix_source_to_target(
            azimuth, elevation, self._pairwise_swap(azimuth), self._pairwise_swap(elevation)
        )
        
        # Apply rotation and then swap the results back. This way
        # if volume contains representations z1, z2 of the same scene, 
        # we get z2', z1' which should match the original z1, z2
        return  self._pairwise_swap(
            eqnr_F.rotate3d(volume, rotation_matrix, mode=self.mode)
        )
    
    def _eval_forward(self, volume: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Eval-specific forward pass supporting both rotation methods.
        
        In evaluation mode, this method supports either:
            1. Direct application of provided rotation matrices
            2. Computing rotation matrices from source-to-target angle parameters
        
        Args:
            volume (torch.Tensor): The 3D volume to rotate. Shape: (batch_size, channels, depth, height, width)
            **kwargs: One of the following sets is required:
                rotation_matrix (torch.Tensor): Batch of rotation matrices. Shape: (batch_size, 3, 3)
                OR
                azimuth_source (torch.Tensor): Source azimuth angles. Shape: (batch_size,)
                elevation_source (torch.Tensor): Source elevation angles. Shape: (batch_size,)
                azimuth_target (torch.Tensor): Target azimuth angles. Shape: (num_views,)
                elevation_target (torch.Tensor): Target elevation angles. Shape: (num_views,)
                
        Returns:
            torch.Tensor: The rotated volume, either with the same shape as input or 
                          expanded to (num_views, channels, depth, height, width) if generating multiple views.
                         
        Raises:
            ValueError: If neither rotation_matrix nor all four angle parameters are provided.
        """
        if all(parm in kwargs for parm in ['azimuth','elevation']):
            if self._pairwise_swap_indices is None or len(self._pairwise_swap_indices) != volume.shape[0]:
                self._pairwise_swap_indices = [idx^1 for idx in range(volume.shape[0])]
            rotation_matrix = eqnr_F.rotation_matrix_source_to_target(
                kwargs['azimuth'], kwargs['elevation'], self._pairwise_swap(kwargs['azimuth']), self._pairwise_swap(kwargs['elevation'])
            )
            return  self._pairwise_swap(
                eqnr_F.rotate3d(volume, rotation_matrix, mode=self.mode)
            )
            
        if 'rotation_matrix' in kwargs and kwargs['rotation_matrix'] is not None:
            return eqnr_F.rotate3d(volume, kwargs['rotation_matrix'], mode=self.mode)
        
        # Check if all angle parameters are provided
        angle_params = ['azimuth_source', 'elevation_source', 'azimuth_target', 'elevation_target']
        if all(param in kwargs for param in angle_params):
            rotation_matrix = eqnr_F.rotation_matrix_source_to_target(
                kwargs['azimuth_source'], kwargs['elevation_source'],
                kwargs['azimuth_target'], kwargs['elevation_target']
            )
            # Check if we are generating new views
            if kwargs['azimuth_source'].shape[0] != kwargs['azimuth_target'].shape[0]:
                num_views = kwargs['azimuth_target'].shape[0]
                volume = volume.expand(num_views, -1, -1, -1, -1)
            return eqnr_F.rotate3d(volume, rotation_matrix, mode=self.mode)
        
        raise ValueError("Either rotation_matrix or all four angle parameters must be provided in eval mode")
    
    def _pairwise_swap(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Swaps elements in a tensor according to _pairwise_swap_indices.
        
        This helper method swaps paired examples in a batch (0↔1, 2↔3, etc.),
        implementing the core mechanism for equivariance training. 
        
        For a batch with consecutive pairs of the same scene from different 
        viewpoints, this operation aligns each scene with its pair's viewpoint.
        
        Args:
            tensor (torch.Tensor): Input tensor to be swapped.
            
        Returns:
            torch.Tensor: Tensor with elements swapped according to _pairwise_swap_indices.
        """
        return tensor[self._pairwise_swap_indices]
    
    
