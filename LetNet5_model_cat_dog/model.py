import torch
import torch.nn as nn
from torchsummary import summary


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2
        )
        self.sig = nn.Sigmoid()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
        )
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(16 * 5 * 5, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sig(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.sig(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.f5(x)
        x = self.sig(x)
        x = self.f6(x)
        x = self.sig(x)
        x = self.f7(x)
        # x = self.softmax(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5(num_classes=10).to(device)
    summary(model, (3, 28, 28))
    # Test the model with a random input
    # x = torch.randn(1, 1, 28, 28).to(device)
    # output = model(x)
    # print("Output shape:", output.shape)
    # x = torch.randn(1, 1, 28, 28).to(device)
    # output = model(x)
    # print("Output shape:", output.shape)
