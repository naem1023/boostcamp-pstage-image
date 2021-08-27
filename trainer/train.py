from tqdm import tqdm

import torch
import wandb
import config
from sklearn.metrics import f1_score
from ray import tune
from .early_stopping import EarlyStopping
from loss_set import get_loss
class BaseTrainer:
    def __init__(
        self, model, epochs, criterion, optimizer, device, batch_size, model_dir, model_name
    ):
        self.model = model
        self.criterion = criterion
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.start_epoch = 1
        self.model_dir = model_dir
        self.model_name = model_name
        self.optimizer = optimizer

    def train(self, train_dataloader, validate_dataloader, feature, epoch):
        train_acc = self._forward(
            train_dataloader, feature=feature, epoch=epoch
        )
        prev_critertion = self.optimizer
        self.criterion = get_loss('crossentropy', cutmix=False)
        valid_acc = self._forward(
            validate_dataloader, train=False, feature=feature, epoch=epoch
        )
        self.criterion = prev_critertion
        return train_acc, valid_acc

    def _forward(
        self, dataloader, train=True, feature=None, epoch=config.NUM_EPOCH, patience=7
    ):
        if train:
            train_tag = 'training'
        else:
            train_tag = 'validation'

        run = wandb.init(
            project="aistage-mask", entity="naem1023", tags=[feature, self.model_name, train_tag]
        )
        wandb.config.learning_rate = config.LEARNING_RATE
        wandb.config.batch_size = config.BATCH_SIZE
        wandb.config.epoch = config.NUM_EPOCH
        wandb.config.k_fold = config.k_split
        wandb.watch(self.model)

        if train:
            self.model.train()
            early_stopping = EarlyStopping(patience=patience, verbose=True, path=self.model_dir, feature=feature,
                                           model_name=self.model_name)
        else:
            self.model.eval()

        for epoch in range(self.epochs):
            running_loss = 0.0
            running_acc = 0.0
            pred_target_list = []
            target_list = []

            with tqdm(dataloader, unit="batch") as tepoch:
                len(dataloader)
                for ind, (images, targets) in enumerate(tepoch):
                    tepoch.set_description(f"{feature}, Epoch {epoch}")
                    images = images.to(self.device)

                    if train and isinstance(targets, (tuple, list)):
                        targets1, targets2, lam = targets
                        targets = (targets1.to(self.device), targets2.to(self.device), lam)
                        target_list += targets[0].tolist()
                        target_list += targets[1].tolist()
                    else:
                        targets = targets.to(self.device)
                        target_list += targets.tolist()

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
                            or config.model_name == "mobilenetv2"
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

                        loss = self.criterion(preds, targets)
                        if train:
                            loss.backward()
                            self.optimizer.step()

                        running_loss += loss.item()

                    pred_target = torch.argmax(preds, dim=1)
                    num = images.size(0)
                    if train and isinstance(targets, (tuple, list)):
                        targets1, targets2, lam = targets
                        correct1 = pred_target.eq(targets1).sum().item()
                        correct2 = pred_target.eq(targets2).sum().item()
                        accuracy = (lam * correct1 + (1 - lam) * correct2) / num
                        pred_target_list += pred_target.tolist()
                        pred_target_list += pred_target.tolist()
                    else:
                        correct_ = pred_target.eq(targets).sum().item()
                        accuracy = correct_ / num
                        # running_correct = (
                        #         torch.sum(pred_label == targets).item() / preds.shape[0]
                        # )

                        # Append inferenced label and real label for f1 score
                        pred_target_list += pred_target.tolist()



                    running_acc += accuracy

                    tepoch.set_postfix(
                        loss=loss.item(), accuracy=accuracy
                    )

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = running_acc / len(dataloader)

            epoch_f1 = f1_score(target_list, pred_target_list, average="macro")
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
            if train:
                # Check loss
                # If loss is decreased, save model.
                early_stopping(epoch_f1, self.model)

                if early_stopping.early_stop:
                    print('Early Stopping!!')
                    break

            print(
                f"epoch-{epoch} Avg Loss: {epoch_loss:.3f}, Avg Accuracy: {epoch_acc:.3f}, f1_score: {epoch_f1:.3f}"
            )

        run.finish()
        return epoch_acc

