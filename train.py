import argparse
import pandas as pd
import numpy as np
import os
from datetime import datetime

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
import torch

from utils import generate_csv

import config

from trainer import feature_train

# fix random seeds for reproducibility
SEED = 12342
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(feature_split, train_df, test_df):
    model_dir = os.path.join(config.model_dir, str(datetime.now().isoformat()))
    os.mkdir(model_dir)
    if feature_split:
        for feature in config.features:
            feature_train(
                train_df, test_df, feature, config.model_name, model_dir
            )


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
