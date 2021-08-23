# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')
import math

import torchvision
import torch


class PretrainedModel:
    """
    Generate pre-trainned model.
    Downlaod model, append layer, and init weight and bias.
    """

    def __init__(self, name, class_num) -> None:
        if name == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=True)
        # imagenet_resnet18 = torchvision.models.resnet50(pretrained=True)

        print("네트워크 필요 입력 채널 개수", self.model.conv1.weight.shape[1])
        print("네트워크 출력 채널 개수 (예측 class type 개수)", self.model.fc.weight.shape[0])
        print(self.model)

        self.model.fc = torch.nn.Linear(
            in_features=512, out_features=class_num, bias=True
        )

        self.init_weight()

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1.0 / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-stdv, stdv)
