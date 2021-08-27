import torch
from torch.nn.functional import embedding
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses

from torchensemble.bagging import BaggingClassifier
from torchensemble.utils.logging import set_logger

import os
import wandb
from datetime import datetime
from utils import transformation
from data_set import MaskDataset
from model import PretrainedModel
from utils import Label
from . import k_fold
import config

from loss_set import FocalLoss


def feature_train(train_df, test_df, feature, model_name, model_dir):
    print(f"{feature}, {model_name}")


    train_dataset = MaskDataset(
        train_df, config.train_dir, transforms=transformation, feature=feature,
    )

    class_num = len(getattr(Label, feature))

    if config.loss == "crossentropy":
        critertion = torch.nn.CrossEntropyLoss()
    elif config.loss == "focal":
        critertion = FocalLoss()
    elif config.loss == "ArcFaceLoss":
        critertion = losses.ArcFaceLoss(
            num_classes=class_num, embedding_size=class_num
        )

    device = torch.device("cuda:0")

    model = PretrainedModel(model_name, class_num).model


    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE
    )  # weight 업데이트를 위한 optimizer를 Adam으로 사용함
    model.to(device)

    if feature == "age":
        epoch = config.NUM_EPOCH + 5
    else:
        epoch = config.NUM_EPOCH

    if config.ensemble:
        ensemble_model = BaggingClassifier(
            estimator=model, n_estimators=10, cuda=True,
        )
        logger = set_logger(f"{feature}-ensemble-training", use_tb_logger=True)
        ensemble_model.set_optimizer("Adam", lr=1e-3, weight_decay=5e-4)

        train_num = int(len(train_dataset) * 0.8)
        test_num = len(train_dataset) - train_num

        train_sampler = torch.utils.data.RandomSampler(
            train_dataset, num_samples=train_num, replacement=True
        )
        test_sampler = torch.utils.data.RandomSampler(
            train_dataset, num_samples=test_num, replacement=True
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=4,
        )
        test_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=test_sampler,
            num_workers=4,
        )

        model_name = f"{model_name}-{feature}-{wandb.run.name}-{datetime.now().isoformat()}.pt"
        model_path = os.path.join(model_dir, model_name)

        ensemble_model.fit(
            train_loader=train_dataloader,
            epochs=epoch,
            test_loader=test_dataloader,
            save_model=True,
            save_dir=model_path,
            log_interval=100
            # wandb=wandb,
        )
    else:
        kt = k_fold.KFoldTrainer(
            config.k_split,
            feature,
            epoch,
            config.BATCH_SIZE,
            model,
            critertion,
            optimizer,
            device,
            model_dir,
            model_name
        )
        valid_acc_list = kt.train(train_dataset)

        # model_name = f"{model_name}-{feature}-{wandb.run.name}-{torch.mean(torch.tensor(valid_acc_list)).item():.2f}-{datetime.now().isoformat()}.pt"
        # torch.save(model.state_dict(), os.path.join(model_dir, model_name))