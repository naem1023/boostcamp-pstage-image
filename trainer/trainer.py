from tqdm import tqdm

import torch


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

    def train(self, train_dataloader, validate_dataloader):
        self._train(train_dataloader)
        self._validatie(validate_dataloader)

    def _train(self, train_dataloader):
        best_test_accuracy = 0.0
        best_test_loss = 9999.0

        for epoch in range(self.epochs):
            running_loss = 0.0
            running_acc = 0.0

            self.model.train()

            with tqdm(train_dataloader, unit="batch") as tepoch:
                # for ind, (images, labels) in enumerate(tepoch):
                for ind, data in enumerate(tepoch):
                    print(data.shape)
                    tepoch.set_description(f"Epoch {epoch}")
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()

                    # grad 계산
                    with torch.set_grad_enabled(True):
                        logits = self.model(images)
                        _, preds = torch.max(logits, 1)

                        loss = self.criterion(logits, labels)

                        loss.backward()
                        self.optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    iter_correct = torch.sum(preds == labels.data)
                    running_acc += iter_correct

                    tepoch.set_postfix(
                        loss=loss.item(),
                        accuracy=iter_correct.item() / self.batch_size,
                    )

            epoch_loss = running_loss / len(self.train_dataloader.dataset)
            epoch_acc = running_acc / len(self.train_dataloader.dataset)

            print(
                f"현재 epoch-{epoch}의 데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}"
            )
        print("학습 종료!")
        print(
            f"최고 accuracy : {best_test_accuracy}, 최고 낮은 loss : {best_test_loss}"
        )

    def _validatie(self, validate_dataloader):
        best_test_accuracy = 0.0
        best_test_loss = 9999.0

        for epoch in range(self.epochs):
            running_loss = 0.0
            running_acc = 0.0

            self.model.eval()

            with tqdm(validate_dataloader, unit="batch") as tepoch:
                for ind, (images, labels) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    logits = self.model(images)
                    _, preds = torch.max(logits, 1)

                    loss = self.criterion(logits, labels)

                    running_loss += loss.item() * images.size(0)
                    iter_correct = torch.sum(preds == labels.data)
                    running_acc += iter_correct

                    tepoch.set_postfix(
                        loss=loss.item(),
                        accuracy=iter_correct.item() / self.batch_size,
                    )

            epoch_loss = running_loss / len(self.train_dataloader.dataset)
            epoch_acc = running_acc / len(self.train_dataloader.dataset)

            print(
                f"현재 epoch-{epoch}의 데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}"
            )
        print("학습 종료!")
        print(
            f"최고 accuracy : {best_test_accuracy}, 최고 낮은 loss : {best_test_loss}"
        )
