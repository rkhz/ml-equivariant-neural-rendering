import torch
from torch.utils.data import Sampler

__all__ = [
    "RandomPairSampler"
]

class RandomPairSampler(Sampler):
    """Samples random elements in pairs. Dataset is assumed to be composed of a
    number of scenes, each rendered in a number of views. This sampler returns
    rendered image in pairs. I.e. for a batch of size 6, it would return e.g.:

    [object 4 - img 5,
     object 4 - img 12,
     object 6 - img 3,
     object 6 - img 19,
     object 52 - img 10,
     object 52 - img 3]


    Arguments:
        dataset (Dataset): Dataset to sample from. This will typically be an
            instance of SceneRenderDataset.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        num_scenes = self.dataset.num_scenes
        num_imgs_per_scene = self.dataset.num_imgs_per_scene

        # Sample num_imgs_per_scene / 2 permutations of the objects
        scene_permutations = [torch.randperm(num_scenes) for _ in range(num_imgs_per_scene // 2)]
        # For each scene, sample a permutation of its images
        img_permutations = [torch.randperm(num_imgs_per_scene) for _ in range(num_scenes)]

        data_permutation = []

        for i, scene_permutation in enumerate(scene_permutations):
            for scene_idx in scene_permutation:
                # Extract image permutation for this object
                img_permutation = img_permutations[scene_idx]
                # Add 2 images of this object to data_permutation
                data_permutation.append(scene_idx.item() * num_imgs_per_scene + img_permutation[2*i].item())
                data_permutation.append(scene_idx.item() * num_imgs_per_scene + img_permutation[2*i + 1].item())

        return iter(data_permutation)

    def __len__(self):
        return len(self.dataset)