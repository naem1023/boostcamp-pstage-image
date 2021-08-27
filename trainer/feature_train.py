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
from loss_set import CutMixCriterion, get_loss

def feature_train(train_df, test_df, feature, model_name, model_dir):
    print(f"{feature}, {model_name}")

    train_dataset = MaskDataset(
        train_df, config.train_dir, transforms=transformation, feature=feature,
    )

    class_num = len(getattr(Label, feature))

    device = torch.device("cuda:0")
    model = PretrainedModel(model_name, class_num).model
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    criterion = get_loss(config.loss, cutmix=True)
    model_config = {
        'class_num': class_num,
        'device': device,
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'k_split': config.k_split,
        'feature': feature,
        'epoch': config.NUM_EPOCH,
        'batch_size': config.BATCH_SIZE,
        'model_dir': model_dir,
        'model_name': model_name,
        'cut_mix': config.cutmix,
        'cut_mix_alpha': config.cutmix_alpha
    }

    kt = k_fold.KFoldTrainer(model_config)
    kt.train(train_dataset)

