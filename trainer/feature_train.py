import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import os

from utils import transformation
from data_set import MaskDataset
from model import PretrainedModel
from utils import Label
from trainer import Trainer
import config


def feature_train(train_df, test_df, feature, model_name):
    train_dataset = MaskDataset(
        train_df, config.train_dir, transforms=transformation, feature=feature,
    )

    critertion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda:0")
    model = PretrainedModel(model_name, len(Label.mask)).model
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

    valid_acc_list = []
    kfold = KFold(n_splits=config.k_split, shuffle=True)
    for fold, (train_idx, validate_idx) in enumerate(
        kfold.split(train_dataset)
    ):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        validate_subsampler = torch.utils.data.SubsetRandomSampler(validate_idx)

        print(f"Start train with {fold} fold")
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=train_subsampler,
            num_workers=4,
        )
        validate_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=validate_subsampler,
            num_workers=4,
        )
        img, labels = next(iter(train_dataloader))

        print(train_dataloader.dataset)
        print(len(train_dataloader.dataset))

        _, valid_acc = trainer.train(train_dataloader, validate_dataloader)
        valid_acc_list.append(valid_acc)

    model_name = f"{model_name}-{feature}-{torch.mean(torch.tensor(valid_acc_list)).item():.3f}.pt"
    torch.save(model.state_dict(), os.path.join(config.model_dir, model_name))
