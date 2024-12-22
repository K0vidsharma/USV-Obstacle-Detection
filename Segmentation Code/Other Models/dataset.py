import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset


class WaterEdgeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, images, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transform
        self.images = images
        self.masks = os.listdir(self.mask_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace("image", "mask"))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255

        if self.transforms:
            augmentations = self.transforms(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
