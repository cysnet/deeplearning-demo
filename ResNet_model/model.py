import torch
from torch import nn
from torchsummary import summary


class Residual(nn.Module):
    def __init__(self, input_channels, output_channels, use_1_1=False, stride=1):
        super().__init__()
        self.ReLu = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=3,
            padding=1,
        )
        self.batchnorm1 = nn.BatchNorm2d(output_channels)
        self.batchnorm2 = nn.BatchNorm2d(output_channels)

        if use_1_1:
            self.conv3 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=1,
                stride=stride,
            )
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.ReLu(self.batchnorm1(self.conv1(x)))
        y = self.batchnorm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.ReLu(y)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            Residual(64, 64),
            Residual(64, 64),
            Residual(64, 128, use_1_1=True, stride=2),
            Residual(128, 128),
            Residual(128, 256, use_1_1=True, stride=2),
            Residual(256, 256),
            Residual(256, 512, use_1_1=True, stride=2),
            Residual(512, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet().to(device)
    summary(model, (1, 224, 224))
