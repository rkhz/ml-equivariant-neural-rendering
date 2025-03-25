import sys
import torch
from torchvision import transforms

from eqnr.utils.data import SceneRenderDataset
from utils.quantitative_evaluation import get_dataset_psnr
from eqnr.nn.models.neural_renderer import NeuralRenderer

# Pick the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get path to experiment folder from command line arguments
if len(sys.argv) != 3:
    raise(RuntimeError("Wrong arguments, use python experiments_psnr.py <model_path> <dataset_folder>"))
model_path = sys.argv[1]
data_dir = sys.argv[2]  # This is usually one of "chairs-test" and "cars-test"

# Load model
model_dict = torch.load(model_path, map_location="cpu")
config = model_dict["config"]
model = NeuralRenderer(
    img_shape=config["img_shape"],
    channels_2d=config["channels_2d"],
    strides_2d=config["strides_2d"],
    channels_3d=config["channels_3d"],
    strides_3d=config["strides_3d"],
    num_channels_inv_projection=config["num_channels_inv_projection"],
    num_channels_projection=config["num_channels_projection"],
    mode=config["mode"]
)
model.load_state_dict(model_dict["state_dict"])
model = model.to(device)

# Initialize dataset
img_transform = transforms.Compose([
    transforms.CenterCrop(128),
    transforms.Resize((3, 128, 128)[1:]),
    transforms.ToTensor()
])
dataset = SceneRenderDataset(path_to_data=data_dir, img_transform=img_transform, allow_odd_num_imgs=True)

# Calculate PSNR
with torch.no_grad():
    psnrs = get_dataset_psnr(device, model, dataset, source_img_idx_shift=64, batch_size=125, max_num_scenes=None)
