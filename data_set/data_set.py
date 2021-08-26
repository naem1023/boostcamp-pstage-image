from torch.utils.data import Dataset, DataLoader
from utils import get_test_img_path
from utils import Label
from PIL import Image
import numpy as np
import torch

import albumentations as A
import albumentations.pytorch


class MaskDataset(Dataset):
    def __init__(
        self, data_df, image_dir, transforms=None, feature=None, train=True
    ):
        self.data_df = data_df
        self.image_dir = image_dir
        self.label = Label()
        if train:
            self.classes = self.label.get_classes(feature)
        self.transforms = transforms
        self.train = train
        self.feature = feature

        if not train:
            # system path list of test images
            self.data_df = get_test_img_path(self.data_df, self.image_dir)

        if self.transforms is None:
            self.transforms = A.Compose(
                [
                    A.CenterCrop(300, 300, p=1),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                    ),
                    albumentations.pytorch.transforms.ToTensorV2(),
                ]
            )

    def __len__(self):
        if isinstance(self.data_df, list):
            return len(self.data_df)
        else:
            return self.data_df.shape[0]

    def __getitem__(self, idx):
        if self.train:
            target_path = self.data_df.iloc[idx]["system_path"]
            label = self.label.get_label(target_path, self.feature)
        else:
            target_path = self.data_df[idx]
            label = target_path

        img = np.array(Image.open(target_path).convert("RGB"))

        if self.transforms:
            img = self.transforms(image=img)
            img = img["image"].type(torch.FloatTensor)

        return img, label
