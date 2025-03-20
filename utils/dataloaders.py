import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from eqnr.utils.data import SceneRenderDataset
from eqnr.utils.data import RandomPairSampler


def scene_render_dataloader(path_to_data='chairs-train', batch_size=16, img_size=(3, 128, 128), crop_size=128):
    """Dataloader for scene render datasets. Returns scene renders in pairs,
    i.e. 1st and 2nd images are of some scene, 3rd and 4th are of some different
    scene and so on.

    Args:
        path_to_data (string): Path to folder containing dataset.
        batch_size (int): Batch size for data.
        img_size (tuple of ints): Size of output images.
        crop_size (int): Size at which to center crop rendered images.

    Notes:
        Batch size must be even.
    """
    assert batch_size % 2 == 0, "Batch size is {} but must be even".format(batch_size)

    img_transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(img_size[1:]),
        transforms.ToTensor()
    ])
    
    dataset = SceneRenderDataset(path_to_data=path_to_data, img_transform=img_transform, allow_odd_num_imgs=False)
    sampler = RandomPairSampler(dataset)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)


def create_batch_from_data_list(data_list):
    """Given a list of datapoints, create a batch.

    Args:
        data_list (list): List of items returned by SceneRenderDataset.
    """
    imgs = []
    azimuths = []
    elevations = []
    for data_item in data_list:
        img, render_params = data_item["img"], data_item["render_params"]
        azimuth, elevation = render_params["azimuth"], render_params["elevation"]
        imgs.append(img.unsqueeze(0))
        azimuths.append(torch.Tensor([azimuth]))
        elevations.append(torch.Tensor([elevation]))
    imgs = torch.cat(imgs, dim=0)
    azimuths = torch.cat(azimuths)
    elevations = torch.cat(elevations)
    return imgs, azimuths, elevations
