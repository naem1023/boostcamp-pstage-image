from tqdm import tqdm

import torch
import wandb
import config
from sklearn.metrics import f1_score
from ray import tune
from .early_stopping import EarlyStopping

class BaseTrainer:
    def __init__(
        self, model, epochs, criterion, optimizer, device, batch_size, model_dir, model_name
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.start_epoch = 1
        self.model_dir = model_dir
        self.model_name = model_name

    def train(self, train_dataloader, validate_dataloader, feature, epoch):
        train_acc = self._forward(
            train_dataloader, feature=feature, epoch=epoch
        )
        valid_acc = self._forward(
            validate_dataloader, train=False, feature=feature, epoch=epoch
        )
        return train_acc, valid_acc

    def _forward(
        self, dataloader, train=True, feature=None, epoch=config.NUM_EPOCH, patience=7
    ):
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=self.model_dir, feature=feature, model_name=self.model_name)
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
                            or config.model_name == "resnet18"
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
                    pred_label_list += pred_label.tolist()
                    label_list += labels.tolist()

                    running_correct = (
                        torch.sum(pred_label == labels).item() / preds.shape[0]
                    )
                    running_acc += running_correct

                    tepoch.set_postfix(
                        loss=loss.item(), accuracy=running_correct
                    )

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            epoch_f1 = f1_score(label_list, pred_label_list, average="micro")
            if config.ray_tune:
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
            # Check loss
            # If loss is decreased, save model.
            early_stopping(epoch_f1, self.model)

            if early_stopping.early_stop:
                print('Early Stopping!!')
                break

            print(
                f"epoch-{epoch} Avg Loss: {epoch_loss:.3f}, Avg Accuracy: {epoch_acc:.3f}, f1_score: {epoch_f1:.3f}"
            )
        return epoch_acc

