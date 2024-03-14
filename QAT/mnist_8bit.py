import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import get_default_qconfig, quantize_qat
from torch.ao.quantization.quantize import quantize, prepare_qat, convert


# Define the model
# 定義模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = torch.nn.ReLU()
        self.conv4 = nn.Conv2d(64, 32, 5, 1)
        self.relu4 = torch.nn.ReLU()
        self.fc1 = nn.Linear(32, 10)
        self.dequant = DeQuantStub()

    def forward(self, x):

        x = self.quant(x)

        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = x.view(-1, 32*1*1)
        x = self.fc1(x)

        x = self.dequant(x)

        return x
    

if __name__ == '__main__':

    # Data loading
    # 數據加載
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = MNIST('../data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Load the pre-trained floating-point model
    # 加载遇訓練的浮點數模型
    float_model_path = '../model/float/checkpoint_10.pth'
    model = Net()
    if float_model_path:
        model.load_state_dict(torch.load(float_model_path))

    # Train the model
    # 訓練模型
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the float model on the test images: {:.2f}%'.format(
        100 * correct / total))


    model.train()
    # Configure quantization
    # 設定量化配置 
    # 'fbgemm' for x86 architecture, 'qnnpack' for ARM
    # 'fbgemm' 適用於 x86 架構, 'qnnpack' 適用於 ARM
    model.qconfig = get_default_qconfig('fbgemm') 

    model = torch.quantization.fuse_modules(model, [['conv1', 'relu1'],
                                                    ['conv2', 'relu2'],
                                                    ['conv3', 'relu3'],
                                                    ['conv4', 'relu4']])

    model = torch.quantization.prepare_qat(model, inplace=True)


    # Quantization-aware training
    # 量化訓練
    for epoch in range(1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Quant Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    # 完成量化訓練後，將模型轉為量化模型
    model.eval()
    model_int8 = torch.quantization.convert(model)


    # 測試量化模型
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model_int8(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print('Accuracy of the quantized model on the test images: {:.2f}%'.format(
        100 * correct / total))
