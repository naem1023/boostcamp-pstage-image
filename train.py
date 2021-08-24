import argparse
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from sklearn.model_selection import KFold

from utils import generate_csv
from data_set import MaskDataset
from model import PretrainedModel
from utils import Label
from trainer import Trainer
import config
from utils import transformation

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(feature_split, train_df, test_df):
    if feature_split:
        # transformation = transforms.Compose([ToTensor(), Resize(224)])

        train_dataset = MaskDataset(
            train_df,
            config.train_dir,
            transforms=transformation,
            feature="mask",
        )
        test_dataset = MaskDataset(
            test_df, config.test_dir, transforms=transformation, train=False
        )
        critertion = torch.nn.CrossEntropyLoss()

        device = torch.device("cuda:0")
        model = PretrainedModel("resnet18", len(Label.mask)).model
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.LEARNING_RATE
        )  # weight 업데이트를 위한 optimizer를 Adam으로 사용함
        model.to(device)
        trainer = Trainer(
            model,
            config.NUM_EPOCH,
            critertion,
            optimizer,
            device,
            config.BATCH_SIZE,
        )

        kfold = KFold(n_splits=8, shuffle=True)
        for fold, (train_idx, validate_idx) in enumerate(
            kfold.split(train_dataset)
        ):
            print(f"Start train with {fold} fold")
            train_dataloader = DataLoader(
                dataset=train_idx,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=4,
            )
            validate_dataloader = DataLoader(
                dataset=validate_idx,
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=4,
            )
            trainer.train(train_dataloader, validate_dataloader)
        # img, labels = next(iter(train_dataloader))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-g-path",
        dest="generate_path",
        action="store_true",
        default=False,
        required=False,
        help="Generate csv file with system path",
    )

    parser.add_argument(
        "-split-train",
        dest="split_train",
        action="store_true",
        default=True,
        required=False,
        help="Train with split features",
    )

    args = parser.parse_args()

    train_df = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv)

    if args.generate_path:
        generate_csv(train_df, config.train_dir, config.with_system_path_csv)
    if args.split_train:
        train_df = pd.read_csv(config.with_system_path_csv)
        main(args.split_train, train_df, test_df)
