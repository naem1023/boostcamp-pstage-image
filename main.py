import argparse
from utils import DataFrameModule
import pandas as pd

test_dir = '/opt/ml/input/data/eval/images'
train_dir = '/opt/ml/input/data/train/images'

test_csv = '/opt/ml/input/data/eval/info.csv'
train_csv = '/opt/ml/input/data/train/train.csv'

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-g-path', dest='generate_path', action='store_true', default=False, \
        required=False, help='Generate csv file with system path')
    
    args = parser.parse_args()

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    if args.generate_path:
        DataFrameModule.generate_csv(train_df, train_dir)

if __name__ == '__main__':
    main()