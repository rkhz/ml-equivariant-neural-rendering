from PIL import Image
from numpy import float32 as np_float32
from torch.utils.data import Dataset


import glob
import json


__all__ = [
    "SceneRenderDataset"
]

class SceneRenderDataset(Dataset):
    """Dataset of rendered scenes and their corresponding camera angles.

    Args:
        path_to_data (string): Path to folder containing dataset.
        img_transform (torchvision.transform): Transforms to be applied to
            images.
        allow_odd_num_imgs (bool): If True, allows datasets with an odd number
            of views. Such a dataset cannot be used for training, since each
            training iteration requires a *pair* of images.

    Notes:
        - Image paths must be of the form "XXXXX.png" where XXXXX are *five*
        integers indexing the image.
        - We assume there are the same number of rendered images for each scene
        and that this number is even.
        - We assume angles are given in degrees.
    """
    def __init__(self, path_to_data='chairs-train', img_transform=None,
                 allow_odd_num_imgs=False):
        self.path_to_data = path_to_data
        self.img_transform = img_transform
        self.allow_odd_num_imgs =  allow_odd_num_imgs
        self.data = []
        # Each folder contains a single scene with different rendering
        # parameters and views
        self.scene_paths = glob.glob(path_to_data + '/*')
        self.scene_paths.sort()  # Ensure consistent ordering of scenes
        self.num_scenes = len(self.scene_paths)
        # Extract number of rendered images per object (which we assume is constant)
        self.num_imgs_per_scene = len(glob.glob(self.scene_paths[0] + '/*.png'))
        # If number of images per scene is not even, drop last image
        if self.num_imgs_per_scene % 2 != 0:
            if not self.allow_odd_num_imgs:
                self.num_imgs_per_scene -= 1
        # For each scene, extract its rendered views and render parameters
        for scene_path in self.scene_paths:
            # Name of folder defines scene name
            scene_name = scene_path.split('/')[-1]

            # Load render parameters
            with open(scene_path + '/render_params.json') as f:
                render_params = json.load(f)

            # Extract path to rendered images of scene
            img_paths = glob.glob(scene_path + '/*.png')
            img_paths.sort()  # Ensure consistent ordering of images
            # Ensure number of image paths is even
            img_paths = img_paths[:self.num_imgs_per_scene]

            for img_path in img_paths:
                # Extract image filename
                img_file = img_path.split('/')[-1]
                # Filenames are of the type "<index>.png", so extract this
                # index to match with render parameters.
                img_idx = img_file.split('.')[0][-5:]  # This should be a string
                # Convert render parameters to float32
                img_params = {key: np_float32(value)
                              for key, value in render_params[img_idx].items()}
                self.data.append({
                    "scene_name": scene_name,
                    "img_path": img_path,
                    "render_params": img_params
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]["img_path"]
        render_params = self.data[idx]["render_params"]

        img = Image.open(img_path)

        # Transform images
        if self.img_transform:
            img = self.img_transform(img)

        # Note some images may contain 4 channels (i.e. RGB + alpha), we only
        # keep RGB channels
        data_item = {
            "img": img[:3],
            "scene_name": self.data[idx]["scene_name"],
            "render_params": self.data[idx]["render_params"]
        }

        return data_item