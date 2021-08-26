from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader
from trainer import BaseTrainer


class KFoldTrainer:
    def __init__(
        self,
        k_split,
        feature,
        epoch,
        batch_size,
        model,
        critertion,
        optimizer,
        device,
        model_dir,
        model_name
    ) -> None:
        self.k_split = k_split
        self.feature = feature
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.trainer = BaseTrainer(
            model, self.epoch, critertion, optimizer, device, self.batch_size, self.model_dir, model_name
        )

    def train(self, train_dataset) -> list:
        valid_acc_list = []
        kfold = KFold(n_splits=self.k_split, shuffle=True)
        for fold, (train_idx, validate_idx) in enumerate(
            kfold.split(train_dataset)
        ):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            validate_subsampler = torch.utils.data.SubsetRandomSampler(
                validate_idx
            )

            print(f"Start train with {fold} fold")
            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                sampler=train_subsampler,
                num_workers=4,
            )
            validate_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                sampler=validate_subsampler,
                num_workers=4,
            )

            _, valid_acc = self.trainer.train(
                train_dataloader, validate_dataloader, self.feature, self.epoch
            )
            valid_acc_list.append(valid_acc)

        return valid_acc_list
