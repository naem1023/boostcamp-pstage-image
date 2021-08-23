import argparse
from utils import DataFrameModule
import pandas as pd
import torch
import numpy as np

test_dir = '/opt/ml/input/data/eval/images'
train_dir = '/opt/ml/input/data/train/images'

test_csv = '/opt/ml/input/data/eval/info.csv'
train_csv = '/opt/ml/input/data/train/train.csv'
with_system_path_csv = '/opt/ml/code/train-with-system-path.csv'

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(feature_split):
    if feature_split:
        data_set 
        data_loader = 
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-g-path', dest='generate_path', action='store_true', default=False, \
        required=False, help='Generate csv file with system path')

    parser.add_argument('-split-train', dest='split_train', action='store_true', default=True, \
        required=False, help='Train with split features')
    
    args = parser.parse_args()

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if args.generate_path:
        DataFrameModule.generate_csv(train_df, train_dir, with_system_path_csv)
    if args.split_train:
        main(feature_split=args.split_train)