from tqdm import tqdm

import torch
import wandb
import config
from sklearn.metrics import f1_score
from ray import tune
from .early_stopping import EarlyStopping
from loss_set import get_loss, CutMixCriterion


class BaseTrainer:
    def __init__(self, model_config):
        self.config = model_config
        self.model = self.config['model']
        self.device = self.config['device']
        self.optimizer = self.config['optimizer']
        self.criterion = self.config['criterion']
        self.scheduler = self.config['scheduler']
        if self.config['cut_mix']:
            self.val_criterion = get_loss('crossentropy', cutmix=False)

    def train(self, train_dataloader, val_dataloader):
        self._forward(train_dataloader, val_dataloader)

    def _forward(self, train_dataloader, val_dataloader, patience=7):
        run = wandb.init(
            project="aistage-mask", entity="naem1023",
            tags=[self.config['feature'], self.config['model_name']]
        )
        wandb.config.learning_rate = config.LEARNING_RATE
        wandb.config.batch_size = config.BATCH_SIZE
        wandb.config.epoch = config.NUM_EPOCH
        wandb.config.k_fold = config.k_split
        wandb.watch(self.model)

        self.model.train()

        early_stopping = EarlyStopping(
            patience=patience, verbose=True, path=self.config['model_dir'], feature=self.config['feature'],
            model_name=self.config['model_name']
        )

        for epoch in range(self.config['epoch']):
            running_loss = 0.0
            running_acc = 0.0
            pred_target_list = []
            target_list = []

            with tqdm(train_dataloader, unit="batch") as tepoch:
                for ind, (images, targets) in enumerate(tepoch):
                    tepoch.set_description(f"{self.config['feature']}: Epoch {epoch}")
                    images = images.to(self.device)

                    # CutMix
                    if isinstance(targets, (tuple, list)):
                        targets1, targets2, lam = targets
                        targets = (targets1.to(self.device), targets2.to(self.device), lam)
                        target_list += targets[0].tolist()
                        target_list += targets[1].tolist()
                    # Normal
                    else:
                        targets = targets.to(self.device)
                        target_list += targets.tolist()

                    self.optimizer.zero_grad()

                    logits = self.model(images)

                    if self.config['model_name'] in ["deit","efficientnet-b4","efficientnet-b7","resnet18","mobilenetv2"]:
                        preds = torch.nn.functional.softmax(logits, dim=-1)
                        # finally get the index of the prediction with highest score
                        # topk_scores, preds = torch.topk(scores, k=1, dim=-1)
                    elif self.config['model_name'] in ["BiT", "volo", "CaiT"]:
                        preds = torch.argmax(logits, dim=1)
                    else:
                        _, preds = torch.max(logits, 1)

                    loss = self.criterion(preds, targets)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    running_loss += loss.item()

                    pred_target = torch.argmax(preds, dim=1)
                    num = images.size(0)
                    if isinstance(targets, (tuple, list)):
                        targets1, targets2, lam = targets
                        correct1 = pred_target.eq(targets1).sum().item()
                        correct2 = pred_target.eq(targets2).sum().item()
                        accuracy = (lam * correct1 + (1 - lam) * correct2) / num
                        pred_target_list += pred_target.tolist()
                        pred_target_list += pred_target.tolist()
                    else:
                        correct_ = pred_target.eq(targets).sum().item()
                        accuracy = correct_ / num
                        # Append inferenced label and real label for f1 score
                        pred_target_list += pred_target.tolist()

                    running_acc += accuracy

            ##################
            # validation
            with torch.no_grad():
                self.model.eval()
                running_val_loss = 0.0
                running_val_acc = 0.0
                val_pred_target_list = []
                val_target_list = []

                with tqdm(val_dataloader, unit="batch") as tepoch:
                    for ind, (images, targets) in enumerate(tepoch):
                        images = images.to(self.device)

                        targets = targets.to(self.device)
                        val_target_list += targets.tolist()

                        logits = self.model(images)

                        if self.config['model_name'] == ["deit", "efficientnet-b4", "efficientnet-b7", "resnet18","mobilenetv2"]:
                            preds = torch.nn.functional.softmax(logits, dim=-1)
                            # finally get the index of the prediction with highest score
                            # topk_scores, preds = torch.topk(scores, k=1, dim=-1)
                        elif self.config['model_name'] == ["BiT", "volo", "CaiT"]:
                            preds = torch.argmax(logits, dim=1)
                        else:
                            _, preds = torch.max(logits, 1)

                        if self.config['cut_mix']:
                            val_loss = self.val_criterion(preds, targets)
                        else:
                            val_loss = self.criterion(preds, targets)

                        running_val_loss += val_loss.item()

                        pred_target = torch.argmax(preds, dim=1)
                        num = images.size(0)
                        correct_ = pred_target.eq(targets).sum().item()
                        accuracy = correct_ / num
                        val_pred_target_list += pred_target.tolist()

                        running_val_acc += accuracy

            tepoch.set_postfix(
                loss=loss.item(), accuracy=accuracy
            )

            epoch_loss = running_loss / len(train_dataloader)
            epoch_acc = running_acc / len(train_dataloader)

            epoch_val_loss = running_val_loss / len(val_dataloader)
            epoch_val_acc = running_val_acc / len(val_dataloader)

            epoch_f1 = f1_score(target_list, pred_target_list, average="macro")
            epoch_val_f1 = f1_score(val_target_list, val_pred_target_list, average="macro")

            if config.ray_tune:
                tune.report(loss=epoch_loss, accuracy=epoch_acc, f1_score=epoch_f1)

            wandb.log({
                "accuracy": epoch_acc,
                "loss": epoch_loss,
                "f1_score": epoch_f1,
                "val_acc": epoch_val_acc,
                "val_loss": epoch_val_loss,
                "val_f1_score": epoch_val_f1,
            })
            # Check loss
            # If loss is decreased, save model.
            early_stopping(epoch_val_f1, self.model)

            if early_stopping.early_stop:
                print('Early Stopping!!')
                break

            print(
                f"epoch-{epoch} val loss: {epoch_val_loss:.3f}, val acc: {epoch_val_acc:.3f}, val_f1_score: {epoch_val_f1:.3f}"
            )

        run.finish()
        return epoch_acc
