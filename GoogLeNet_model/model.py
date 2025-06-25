import torch
from torch import nn
from torchsummary import summary


class Inception(nn.Module):
    def __init__(
        self, in_channels, out1x1, out3x3red, out3x3, out5x5red, out5x5, pool_proj
    ):
        super(Inception, self).__init__()
        self.ReLu = nn.ReLU()

        self.branch1x1 = nn.Conv2d(in_channels, out1x1, kernel_size=1)

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out3x3red, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out3x3red, out3x3, kernel_size=3, padding=1),
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out5x5red, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out5x5red, out5x5, kernel_size=5, padding=2),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        p1 = self.ReLu(self.branch1x1(x))
        p2 = self.ReLu(self.branch3x3(x))
        p3 = self.ReLu(self.branch5x5(x))
        p4 = self.ReLu(self.branch_pool(x))
        outputs = [p1, p2, p3, p4]
        return torch.cat(outputs, 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3), nn.ReLU(), nn.MaxPool2d(3, 2, 1)
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )

        self.b3 = nn.Sequential(
            Inception(192, 64, 96, 128, 16, 32, 32),
            Inception(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2, 1),
        )

        self.b4 = nn.Sequential(
            Inception(480, 192, 96, 208, 16, 48, 64),
            Inception(512, 160, 112, 224, 24, 64, 64),
            Inception(512, 128, 128, 256, 24, 64, 64),
            Inception(512, 112, 128, 288, 32, 64, 64),
            Inception(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2, 1),
        )

        self.b5 = nn.Sequential(
            Inception(832, 256, 160, 320, 32, 128, 128),
            Inception(832, 384, 192, 384, 48, 128, 128),
        )

        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)

        x = self.b6(x)

        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GoogLeNet().to(device=device)

    print(summary(model, (1, 224, 224)))
