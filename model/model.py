# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')
import torchvision
# imagenet_resnet18 = torchvision.models.resnet50(pretrained=True)
model = torchvision.models.resnet18(pretrained=True)
print("네트워크 필요 입력 채널 개수", model.conv1.weight.shape[1])
print("네트워크 출력 채널 개수 (예측 class type 개수)", model.fc.weight.shape[0])
print(model)

MASK_CLASS_NUM = 18
import math
model.fc = torch.nn.Linear(in_features=512, out_features=MASK_CLASS_NUM, bias=True)
torch.nn.init.xavier_uniform_(model.fc.weight)
stdv = 1. / math.sqrt(model.fc.weight.size(1))
model.fc.bias.data.uniform_(-stdv, stdv)