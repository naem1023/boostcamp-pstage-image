import pandas as pd
from tqdm import tqdm
from data_set import MaskDataset
import config


def main():
    train_df = pd.read_csv(config.with_system_path_csv)
    train_set = []
    count = []
    for feature in config.features:
        train_set.append(
            (feature, MaskDataset(train_df, config.train_dir, feature=feature))
        )

    for dataset in train_set:
        count_temp = [0 for _ in range(len(dataset[1].classes))]
        for data in tqdm(dataset[1]):
            count_temp[data[1]] += 1
        count.append((dataset[0], count_temp))

    print(count)

    print([sum(c[1]) for c in count])


if __name__ == "__main__":
    main()
