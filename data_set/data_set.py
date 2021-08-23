from torch.utils.data import Dataset, DataLoader
from utils import get_test_img_path
from utils import Label
from PIL import Image


class MaskDataset(Dataset):
    def __init__(
        self, data_df, image_dir, transforms=None, feature=None, train=True
    ):
        self.data_df = data_df
        self.image_dir = image_dir
        self.label = Label()
        self.classes = self.label.get_classes(feature)
        self.transforms = transforms
        self.train = train
        self.feature = feature

        if not train:
            # system path list of test images
            self.data_df = get_test_img_path(self.data_df, self.image_dir)

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, idx):
        if self.train:
            target_path = self.data_df.iloc[idx]["system_path"]
            label = self.label.get_label(target_path, self.feature)
        else:
            target_path = self.data_df[idx]
            label = None

        img = Image.open(target_path)

        if self.transforms:
            img = self.transforms(img)

        return img, label
