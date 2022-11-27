import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
# from torchvision.utils import save_image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class WaterDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return list(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path).convert("RGB"))
        split_half = int(image.shape[1]/2)
        input_image = image[:, :split_half, :]
        target_image = image[:, :split_half, :]
        return input_image, target_image
