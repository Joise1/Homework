import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import os

# 数据预处理
transform_train = transforms.Compose([
     transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),   # 数据增强，为了防止出现数据过拟合
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

transform_test = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# 定义神经网络
class RNet(nn.Module):

    def __init__(self):
        super(RNet, self).__init__()
        self.lstm = nn.LSTM(3*32, 128, num_layers=3,
                            batch_first=True)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.classifier(x)

        return x


# 训练神经网络
def train(network, dev):
    # 定义优化器和损失函数
    path = 'weights.tar'
    init_epoch = 0
    optimizer = optim.Adam(network.parameters(), lr=0.0001)
    if os.path.exists(path) is not True:
        loss = nn.CrossEntropyLoss()
    else:  # 读取中断模型结果，继续运行模型
        checkpoint = torch.load(path)
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        init_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    # 训练
    for epoch in range(init_epoch, 100):
        time_start = time.time()  # 计时工具，每个epoch训练用时
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            # 数据读取
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev)
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播 + 后向传播 + 求损失
            outputs = network(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
            # 输出运行结果
            running_loss += l.item()
            if i % 500 == 499:  # 每500个mini-batch输出一次
                print('[%d, %5d] loss: %.4f' % (epoch, i, running_loss / 500))
                running_loss = 0.0
                _, predicted = torch.max(outputs.data, 1)
                torch.save({'epoch': epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss
                            }, path)

        print('epoch %d 用时 %3f s' % (epoch, time.time() - time_start))

    print('训练结束')


# 测试神经网络
def test(network, dev):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(dev), labels.to(dev)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('神经网络对于10000张测试图像的测试精度如下: %.3f %%' % (100.0 * correct / total))


if __name__ == '__main__':
    # 0. 参数定义
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 1. 读取并且预处理数据集
    # 训练集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
    # 测试集
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)
    # 2. 定义神经网络
    net = RNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    # 3. 神经网络训练
    train(net, device)
    # 4. 神经网络测试
    test(net, device)