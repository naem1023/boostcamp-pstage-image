import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from itertools import product

from trainer import Trainer
from data_set import MaskDataset
from utils import transformation
from model import PretrainedModel
from predict import Predictor
from utils import Label

import config
import glob


def main():
<<<<<<< HEAD
    model_path = glob.glob(config.model_dir + f"/{config.model_name}*.pt")
    print(model_path)
    test_df = pd.read_csv(config.test_csv)

    result_list = []
    for feature in config.features:
        for path in model_path:
            if feature in path:
                break
=======
    model_path = glob.glob(config.model_dir + "/*.pt")
    test_df = pd.read_csv(config.test_csv)

    result_list = []
    for path in model_path:
        feature = path.split("-")[0]
>>>>>>> 6852a782b6c0b56e054b91befbdaeffc962a878d
        test_dataset = MaskDataset(
            test_df, config.test_dir, transforms=transformation, train=False
        )

        test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=config.BATCH_SIZE, num_workers=4,
        )

        device = torch.device("cuda:0")
<<<<<<< HEAD
        model = PretrainedModel(config.model_name, len(Label.mask)).model
=======
        model = PretrainedModel("resnet18", len(Label.mask)).model
>>>>>>> 6852a782b6c0b56e054b91befbdaeffc962a878d
        model.load_state_dict(torch.load(path))

        model.to(device)
        predictor = Predictor(
            model, config.NUM_EPOCH, device, config.BATCH_SIZE,
        )

        result = predictor.predict(test_dataloader, feature)
        result_list.append(result)

    predict(result_list)


def predict(result):
    """
    result row
        0: age
        1: mask
        2: gender
    """
    mask = [0, 1, 2]
    gender = [0, 1]
    age = [0, 1, 2]

    label_number = list(product(mask, gender, age))
    print(label_number)

    submission = []
    for i in range(len(result[0])):
        path = result[0][i][0]
        pred_class = label_number.index(
            (result[0][i][1], result[1][i][1], result[2][i][1])
        )
        submission.append([path, pred_class])
    result_df = pd.DataFrame.from_records(
        submission, columns=["ImageID", "ans"]
    )
    result_df.to_csv("submission.csv", index=False)
<<<<<<< HEAD
    result_df.to_csv("submission.csv", index=False)
=======
>>>>>>> 6852a782b6c0b56e054b91befbdaeffc962a878d


if __name__ == "__main__":
    main()
