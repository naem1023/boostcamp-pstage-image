from sys import path
from tqdm import tqdm

import torch


class Predictor:
    def __init__(
        self, model, epochs, device, batch_size,
    ):
        self.model = model
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.start_epoch = 1

    def predict(self, dataloader, feature):
        result = []

<<<<<<< HEAD
=======
        running_loss = 0.0
        running_acc = 0.0

>>>>>>> 6852a782b6c0b56e054b91befbdaeffc962a878d
        self.model.eval()

        with tqdm(dataloader, unit="batch") as tepoch:
            len(dataloader)
            for ind, (images, paths) in enumerate(tepoch):
                tepoch.set_description(f"{feature}")
                images = images["image"].type(torch.FloatTensor).to(self.device)
                with torch.no_grad():
                    logits = self.model(images)
                    _, preds = torch.max(logits, 1)

                    path_list = [path.split("/")[-1] for path in paths]

                    result.append([path_list, preds.tolist()])

        merge_result = []
        for epoch_pred in result:
            for path, pred_class in zip(epoch_pred[0], epoch_pred[1]):
                merge_result.append([path, pred_class])

        return merge_result

