from tqdm import tqdm

import torch


class Trainer:
    def __init__(
        self,
        model,
        epochs,
        criterion,
        optimizer,
        train_dataloader,
        device,
        batch_size,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.device = device
        self.batch_size = batch_size
        self.start_epoch = 1
        # self.checkpoint_dir = config.save_dir

    def train(self):
        self._train_epoch()

    def _train_epoch(self):
        best_test_accuracy = 0.0
        best_test_loss = 9999.0

        for epoch in range(self.epochs):
            running_loss = 0.0
            running_acc = 0.0

            self.model.train()  # 네트워크 모델을 train 모드로 두어 gradient을 계산하고, 여러 sub module (배치 정규화, 드롭아웃 등)이 train mode로 작동할 수 있도록 함

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for ind, (images, labels) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    # (참고.해보기) 현재 tqdm으로 출력되는 것이 단순히 진행 상황 뿐인데 현재 epoch, running_loss와 running_acc을 출력하려면 어떻게 할 수 있는지 tqdm 문서를 보고 해봅시다!
                    # hint - with, pbar
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()  # parameter gradient를 업데이트 전 초기화함

                    with torch.set_grad_enabled(
                        True
                    ):  # train 모드일 시에는 gradient를 계산하고, 아닐 때는 gradient를 계산하지 않아 연산량 최소화
                        logits = self.model(images)
                        # 모델에서 linear 값으로 나오는 예측 값 ([0.9,1.2, 3.2,0.1,-0.1,...])을 최대 output index를 찾아 예측 레이블([2])로 변경함
                        _, preds = torch.max(logits, 1)

                        loss = self.criterion(logits, labels)

                        loss.backward()  # 모델의 예측 값과 실제 값의 CrossEntropy 차이를 통해 gradient 계산
                        self.optimizer.step()  # 계산된 gradient를 가지고 모델 업데이트

                    running_loss += loss.item() * images.size(
                        0
                    )  # 한 Batch에서의 loss 값 저장
                    iter_correct = torch.sum(preds == labels.data)
                    running_acc += iter_correct  # 한 Batch에서의 Accuracy 값 저장

                    tepoch.set_postfix(
                        loss=loss.item(),
                        accuracy=iter_correct.item() / self.batch_size,
                    )

            # 한 epoch이 모두 종료되었을 때,
            epoch_loss = running_loss / len(self.train_dataloader.dataset)
            epoch_acc = running_acc / len(self.train_dataloader.dataset)

            print(
                f"현재 epoch-{epoch}의 데이터 셋에서 평균 Loss : {epoch_loss:.3f}, 평균 Accuracy : {epoch_acc:.3f}"
            )
        print("학습 종료!")
        print(
            f"최고 accuracy : {best_test_accuracy}, 최고 낮은 loss : {best_test_loss}"
        )
