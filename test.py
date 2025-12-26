import os

import torch
import torchvision.transforms
from PIL import Image
from torch import nn

class Mynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Flatten(),   #展平
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.Linear(in_features=120, out_features=84),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


root_dir='test_number'
number_name='0409-8.png'

img_path = os.path.join(root_dir, number_name)
#拼接，形成路径
img = Image.open(img_path)
# 读出图片 img.show()
img_1 = img.convert('L')#将三通道转换成单通道


tran_pose = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32,32)),
    torchvision.transforms.ToTensor()
])

mynet = torch.load('MNIST4.9_20_acc_0.9818999767303467.pth',map_location=torch.device('cpu'),weights_only=False)
#将显卡跑出的模型，用cpu测试


img_1 = tran_pose(img_1)
print(img_1.shape)#把图片转换成了3*32*32  三通道还需要转换成1通道
img_1 = torch.reshape(img_1, (1,1,32,32))
print(img_1.shape)

output = mynet(img_1)

mynet.eval()
with torch.no_grad():
    output = mynet(img_1)
    number = output.argmax(axis=1).item()
    print(f'识别的数字是{number}')
