from PIL import Image
import math
import cv2
import numpy as np
import albumentations as A
import os
import torch
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
import torch.nn.functional as F


def split_image_crops(directory, model, kernel_size=256, device="cpu"):
    model = model.to(device)
    model.eval()

    for idx, image_file in enumerate(os.listdir(directory)):
        image = Image.open(os.path.join(directory, image_file)).convert("RGB")
        width, height = image.size
        max_size = math.cell(max(width, height)/kernel_size)*kernel_size
        pad_height = max_size - height
        pad_width = max_size - width

        image = np.array(image)
        augment = A.Compose([
            A.PadIfNeeded(min_width=max_size, min_height=max_size,
                          border_mode=cv2.BORDER_REFLECT),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(
                0.5, 0.5, 0.5), max_pixel_value=255.0),
            ToTensorV2()
        ])
        image = augment(image=image)['image'].to(device)
        img_size = image.shape[2]
        image = image.permute(1, 2, 0)
        kh, kw = kernel_size, kernel_size
        dh, dw = 32, 32
        patches = image.unfold(0, kh, dh).unfold(1, kw, dw)
        patches = patches.contigous().view(-1, 3, kh, kw)
        with torch.no_grad():
            batch_size = 32
            for id in tqdm(range(math.ceil(patches.shape[0]/batch_size))):
                from_idx = id*batch_size
                to_idx = min((id+1)*batch_size, patches.shape[0])

                curr_patch = patches[from_idx:to_idx].to(device)
                patch = model(curr_patch)
                patches[from_idx:to_idx] = (patch*0.5+0.5).to("cpu")
        patches = patches.view(
            1, patches.shape[0], 3*kernel_size*kernel_size).permute(0, 2, 1)
        output = F.fold(patches, output_size=(img_size, img_size),
                        kernel_size=kernel_size, stride=dh)
        recovery_mask = F.fold(torch.ones_like(patches), output_size=(
            img_size, img_size), kernel_size=kernel_size, stride=dh)
        output /= recovery_mask
        augment_back = A.Compose([
            A.CenterCrop(height=max_size-int(pad_height),
                         width=max_size-int(pad_width)),
            ToTensorV2
        ])
        x = augment_back(image=output.squeeze(
            0).detach().cpu().permute(1, 2, 0).numpy())['image']
        save_image(x, f"saved/test_results_{idx}.png")

    model.train()
