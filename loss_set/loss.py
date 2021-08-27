import config

import torch
from data_loader import CutMixCriterion
from loss_set import FocalLoss
from pytorch_metric_learning import losses

def get_loss(name, cutmix):
    if cutmix:
        critertion = CutMixCriterion(reduction='mean')
    elif name == "crossentropy":
        critertion = torch.nn.CrossEntropyLoss()
    elif name == "focal":
        critertion = FocalLoss()
    elif name == "ArcFaceLoss":
        critertion = losses.ArcFaceLoss(
            num_classes=config.class_num, embedding_size=config.class_num
        )
    return critertion