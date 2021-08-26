from tqdm import tqdm

import torch
import wandb
import config
from sklearn.metrics import f1_score
from ray import tune


class BaseTrainer:
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

    def train(self, train_dataloader, validate_dataloader, feature, epoch):
        train_acc = self._forward(
            train_dataloader, feature=feature, epoch=epoch
        )
        valid_acc = self._forward(
            validate_dataloader, train=False, feature=feature, epoch=epoch
        )
        return train_acc, valid_acc

    def _forward(
        self, dataloader, train=True, feature=None, epoch=config.NUM_EPOCH
    ):
        for epoch in range(self.epochs):
            running_loss = 0.0
            running_acc = 0.0
            pred_label_list = []
            label_list = []

            if train:
                self.model.train()
            else:
                self.model.eval()

            with tqdm(dataloader, unit="batch") as tepoch:
                len(dataloader)
                for ind, (images, labels) in enumerate(tepoch):
                    tepoch.set_description(f"{feature}, Epoch {epoch}")
                    images = images.to(self.device)

                    labels = labels.to(self.device)

                    if train:
                        self.optimizer.zero_grad()

                    # grad 계산
                    with torch.set_grad_enabled(train):
                        logits = self.model(images)
                        if (
                            config.model_name == "deit"
                            or config.model_name == "efficientnet-b4"
                            or config.model_name == "efficientnet-b7"
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
                    pred_label = torch.argmax(preds, dim=1)

                    # Append inferenced label and real label for f1 score
                    pred_label_list.extend(pred_label.item())
                    label_list.extend(labels)

                    running_correct = (
                        torch.sum(pred_label == labels).item() / preds.shape[0]
                    )
                    running_acc += running_correct

                    tepoch.set_postfix(
                        loss=loss.item(), accuracy=running_correct,
                    )

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            epoch_f1 = f1_score(label_list, pred_label_list, average="micro")
            tune.report(loss=epoch_loss, accuracy=epoch_acc, f1_score=epoch_f1)

            if train:
                wandb.log(
                    {
                        "accuracy": epoch_acc,
                        "loss": epoch_loss,
                        "f1_score": epoch_f1,
                    }
                )
            else:
                wandb.log(
                    {
                        "val_acc": epoch_acc,
                        "val_loss": epoch_loss,
                        "val_f1_score": epoch_f1,
                    }
                )

            print(
                f"현재 epoch-{epoch}의 데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}"
            )
        return epoch_acc

