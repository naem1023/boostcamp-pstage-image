import argparse
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize

from utils import DataFrameModule
from data_set import MaskDataset
from model import PretrainedModel
from utils import Label
from trainer import Trainer

test_dir = "/opt/ml/input/data/eval/images"
train_dir = "/opt/ml/input/data/train/images"

test_csv = "/opt/ml/input/data/eval/info.csv"
train_csv = "/opt/ml/input/data/train/train.csv"
with_system_path_csv = "/opt/ml/code/train-with-system-path.csv"

features = ["age", "mask", "gender"]
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCH = 5


def main(feature_split, train_df, test_df):
    if feature_split:
        transformation = transforms.Compose([ToTensor(), Resize(224)])

        train_set = MaskDataset(
            train_df, train_dir, transforms=transformation, feature="mask"
        )
        test_set = MaskDataset(
            test_df, test_dir, transforms=transformation, train=False
        )
        train_dataloader = DataLoader(
            dataset=train_set, batch_size=BATCH_SIZE, shuffle=True
        )
        test_dataloader = DataLoader(
            dataset=test_set, batch_size=BATCH_SIZE, shuffle=False
        )
        img, labels = next(iter(train_dataloader))

        critertion = (
            torch.nn.CrossEntropyLoss()
        )  # 분류 학습 때 많이 사용되는 Cross entropy loss를 objective function으로 사용 - https://en.wikipedia.org/wiki/Cross_entropy

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = PretrainedModel("resnet18", len(Label.mask)).model
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE
        )  # weight 업데이트를 위한 optimizer를 Adam으로 사용함
        model.to(device)

        trainer = Trainer(
            model,
            NUM_EPOCH,
            critertion,
            optimizer,
            train_dataloader,
            device,
            BATCH_SIZE,
        )

        trainer.train()


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

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if args.generate_path:
        DataFrameModule.generate_csv(train_df, train_dir, with_system_path_csv)
    if args.split_train:
        train_df = pd.read_csv(with_system_path_csv)
        main(args.split_train, train_df, test_df)
