# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')
import math

import torchvision
import torch

from efficientnet_pytorch import EfficientNet
from vit_pytorch import ViT
from vit_pytorch.deepvit import DeepViT
from vit_pytorch.cait import CaiT
import timm
from model import volo
from tlt.utils import load_pretrained_weights


class PretrainedModel:
    """
    Generate pre-trainned model.
    Downlaod model, append layer, and init weight and bias.
    """

    def __init__(self, name, class_num) -> None:
        if name == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=True)
            self.model.fc = torch.nn.Linear(
                in_features=512, out_features=class_num, bias=True
            )
            print("네트워크 필요 입력 채널 개수", self.model.conv1.weight.shape[1])
            print(
                "네트워크 출력 채널 개수 (예측 class type 개수)",
                self.model.fc.weight.shape[0],
            )

            self.init_weight()

        elif name == "efficientnet":
            self.model = EfficientNet.from_pretrained(
                "efficientnet-b7", num_classes=class_num
            )
            self.model.fc = torch.nn.Linear(
                in_features=512, out_features=class_num, bias=True
            )
        elif name == "volod3":
            self.model = volo.volo_d1()
            load_pretrained_weights(
                model=self.model,
                checkpoint_path="/opt/ml/downloads/d1_224_84.2.pth.tar",
                use_ema=False,
                strict=False,
                num_classes=class_num,
            )
        elif name == "BiT":
            self.model = timm.create_model(
                "resnetv2_101x1_bitm", pretrained=True, num_classes=class_num,
            )
        elif name == "CaiT":
            # https://github.com/lucidrains/vit-pytorch
            self.model = CaiT(
                image_size=224,
                patch_size=32,
                num_classes=class_num,
                dim=1024,
                depth=12,  # depth of transformer for patch to patch attention only
                cls_depth=2,  # depth of cross attention of CLS tokens to patch
                heads=16,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1,
                layer_dropout=0.05,  # randomly dropout 5% of the layers
            )

    def init_weight(self):
        torch.nn.init.xavier_uniform_(self.model.fc.weight)
        stdv = 1.0 / math.sqrt(self.model.fc.weight.size(1))
        self.model.fc.bias.data.uniform_(-stdv, stdv)
