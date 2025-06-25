import os
import sys

sys.path.append(os.getcwd())

import torch  # 导入PyTorch主库
from torch import nn  # 从torch中导入神经网络模块
from torchsummary import summary  # 导入torchsummary用于模型结构总结
import torch.nn.functional as F  # 导入PyTorch的函数式API，常用于激活函数、dropout等


class AlexNet(nn.Module):  # 定义AlexNet模型，继承自nn.Module
    def __init__(self):  # 构造函数，初始化网络结构
        super(AlexNet, self).__init__()  # 调用父类的构造函数
        self.ReLU = nn.ReLU()  # 定义ReLU激活函数，后续多次复用
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=96, stride=4, kernel_size=11
        )  # 第一层卷积，输入通道1，输出通道96，步幅4，卷积核11x11
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)  # 第一层池化，3x3窗口，步幅2

        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, stride=1, kernel_size=5, padding=2
        )  # 第二层卷积，输入96通道，输出256通道，5x5卷积核，padding=2
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)  # 第二层池化，3x3窗口，步幅2

        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, stride=1, kernel_size=3, padding=1
        )  # 第三层卷积，输入256通道，输出384通道，3x3卷积核，padding=1
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384, stride=1, kernel_size=3, padding=1
        )  # 第四层卷积，输入384通道，输出384通道，3x3卷积核，padding=1
        self.conv5 = nn.Conv2d(
            in_channels=384, out_channels=256, stride=1, kernel_size=3, padding=1
        )  # 第五层卷积，输入384通道，输出256通道，3x3卷积核，padding=1

        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)  # 第三层池化，3x3窗口，步幅2
        self.flatten = nn.Flatten()  # 展平层，将多维输入展平成一维

        self.fc1 = nn.Linear(
            in_features=256 * 6 * 6, out_features=4096
        )  # 第一个全连接层，输入256*6*6，输出4096
        self.fc2 = nn.Linear(
            in_features=4096, out_features=4096
        )  # 第二个全连接层，输入4096，输出4096
        self.fc3 = nn.Linear(
            in_features=4096, out_features=10
        )  # 第三个全连接层，输入4096，输出10（假设10分类）

    def forward(self, x):  # 定义前向传播过程
        x = self.conv1(x)  # 输入通过第一层卷积
        x = self.ReLU(x)  # 激活
        x = self.pool1(x)  # 池化

        x = self.conv2(x)  # 第二层卷积
        x = self.ReLU(x)  # 激活
        x = self.pool2(x)  # 池化

        x = self.conv3(x)  # 第三层卷积
        x = self.ReLU(x)  # 激活
        x = self.conv4(x)  # 第四层卷积
        x = self.ReLU(x)  # 激活
        x = self.conv5(x)  # 第五层卷积
        x = self.ReLU(x)  # 激活
        x = self.pool3(x)  # 池化

        x = self.flatten(x)  # 展平为一维向量

        x = self.fc1(x)  # 第一个全连接层
        x = self.ReLU(x)  # 激活
        x = F.dropout(x, p=0.5)  # dropout防止过拟合，丢弃概率0.5

        x = self.fc2(x)  # 第二个全连接层
        x = self.ReLU(x)  # 激活
        x = F.dropout(x, p=0.5)  # dropout防止过拟合，丢弃概率0.5

        x = self.fc3(x)  # 第三个全连接层，输出最终结果
        return x  # 返回输出


if __name__ == "__main__":  # 如果作为主程序运行
    model = AlexNet()  # 实例化AlexNet模型
    print(model)  # 打印模型结构
    summary(
        model, input_size=(1, 227, 227), device="cpu"
    )  # 打印模型摘要，输入尺寸为(1, 227, 227)，单通道
    # Note: The input size is set to (1, 224, 224) for a single-channel image.
    # Adjust the input size according to your dataset if necessary.
