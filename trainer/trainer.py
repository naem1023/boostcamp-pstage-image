from tqdm import tqdm

import torch
import wandb
import config


class Trainer:
    def __init__(
        self, model, epochs, criterion, optimizer, device, batch_size,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.start_epoch = 1

    def train(self, train_dataloader, validate_dataloader, feature):
        train_acc = self._forward(train_dataloader, feature=feature)
        valid_acc = self._forward(
            validate_dataloader, train=False, feature=feature
        )
        return train_acc, valid_acc

    def _forward(self, dataloader, train=True, feature=None):
        for epoch in range(self.epochs):
            running_loss = 0.0
            running_acc = 0.0

            if train:
                self.model.train()
            else:
                self.model.eval()

            with tqdm(dataloader, unit="batch") as tepoch:
                len(dataloader)
                for ind, (images, labels) in enumerate(tepoch):
                    tepoch.set_description(f"{feature}, Epoch {epoch}")
                    images = (
                        images["image"].type(torch.FloatTensor).to(self.device)
                    )

                    labels = labels.to(self.device)

                    if train:
                        self.optimizer.zero_grad()

                    # grad 계산
                    with torch.set_grad_enabled(train):
                        logits = self.model(images)
                        if (
                            config.model_name == "deit"
                            or config.model_name == "efficientnet"
                        ):
                            preds = torch.nn.functional.softmax(logits, dim=-1)
                            # finally get the index of the prediction with highest score
                            # topk_scores, preds = torch.topk(scores, k=1, dim=-1)
                        elif (
                            config.model_name == "BiT"
                            or config.model_name == "volo"
                            or config.model_name == "CaiT"
                        ):
                            preds = torch.argmax(logits, dim=1)
                        else:
                            _, preds = torch.max(logits, 1)

                        loss = self.criterion(preds, labels)

                        if train:
                            loss.backward()
                            self.optimizer.step()

                    # running_loss += loss.item() * images.size(0)
                    running_loss += loss.item()
                    running_correct = (
                        torch.sum(torch.argmax(preds, dim=1) == labels).item()
                        / preds.shape[0]
                    )
                    running_acc += running_correct

                    tepoch.set_postfix(
                        loss=loss.item(), accuracy=running_correct,
                    )

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            if train:
                wandb.log({"accuracy": epoch_acc})
                wandb.log({"loss": epoch_loss})
            else:
                wandb.log({"val_acc": epoch_acc})
                wandb.log({"val_loss": epoch_loss})

            print(
                f"현재 epoch-{epoch}의 데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}"
            )
        return epoch_acc

