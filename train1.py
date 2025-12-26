
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data_set = torchvision.datasets.MNIST(root='dataset', train=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32,32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
]),download=True)

test_data_set=torchvision.datasets.MNIST(root='dataset', train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32,32)),
    torchvision.transforms.ToTensor()
]),download=True)   #导入数据集

train_data_size=len(train_data_set)
test_data_size=len(test_data_set)

print(f'训练集长度为{train_data_size}')
print(f'测试集长度为{test_data_size}')

train_data_loader = DataLoader(dataset=train_data_set, batch_size=64, shuffle=True,drop_last=True)
test_data_loader = DataLoader(dataset=test_data_set, batch_size=64, shuffle=True,drop_last=True)


#构建网络模型，重点
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

mynet = Mynet()
mynet = mynet.to(device)#网络加速

#定义损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)#损失函数加速

#定义优化器
learning_rate = 0.01
optim = torch.optim.SGD(mynet.parameters(), learning_rate)

train_step = 0

epoch = 40

if __name__ == '__main__':
    for i in range(epoch):
        print(f'----------第{i+1}轮训练-----------')
        mynet.train()
        for data in train_data_loader:

            imgs,targets=data
            imgs = imgs.to(device)
            targets = targets.to(device)#第三次加速
            outputs=mynet(imgs)

            loss = loss_fn(outputs, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()

            train_step += 1
            if train_step % 100 == 0:
                print(f'第{train_step}次训练,loss={loss.item()}')

        mynet.eval()
        accuracy=0
        total_accuracy=0
        with torch.no_grad():
            for data in test_data_loader:
                imgs,targets=data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs=mynet(imgs)

                accuracy = (outputs.argmax(axis=1) == targets).sum()
                total_accuracy = total_accuracy + accuracy
            print(f'{i+1}轮训练结束，准确率{total_accuracy/test_data_size}') #3.9日晚上21.00
            torch.save(mynet,f'MNIST4.9_{i}_acc_{total_accuracy/test_data_size}.pth')#保存
















