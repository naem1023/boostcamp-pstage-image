from torch.utils.data import Dataset, DataLoader

class MaskDataset(Dataset):
    def __init__(self, data_df, images_dir, transforms, feature=None, train=True):    
        self.data_df = data_df
        self.images_dir = images_dir
        self.classes = range(18)
        self.transforms = transforms
        self.train = traind
        self.feature = feature

        if not train:
            self.test_path_list = get_test_img_path()
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if train:
            # merge path and feature
            base_path = self.data_df.loc[idx, 'path']

            # Get all possilbe path for base_path and feature
            target_path = get_train_img_path(self.images_dir, base_path, self.feature)

            # Append asterisk for using glob, cause all the images have different extension.
            if isinstance(target_path, list):
                target_path = [target_path + '*' for p in target_path]
            elif isinstance(target_path, str):
                target_path = target_path + '*'
            target_path = glob.glob(target_path)
        else:
            target_path = self.test_path_list[idx]

        img = Image.open(target_path)

        if self.train:
            label = get_label(path)
        else:
            label = get_test_label(path)

        if self.transforms:
            img = self.transforms(img)
            
        return img, label