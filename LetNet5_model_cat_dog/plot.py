from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


train_dataset = FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
)

tra_loader = DataLoader(
    dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0
)


for step, (x, y) in enumerate(tra_loader):
    if step > 0:
        break

    batch_x = x.squeeze().numpy()
    batch_y = y.numpy()
    class_label = train_dataset.classes
    print("batch_x:", batch_x)
    print("batch_y:", batch_y)
    print("batch_y count:", len(batch_y))
    print("class_label:", class_label)


plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap="gray")
    plt.title(class_label[batch_y[ii]], size=10)
plt.axis("off")
plt.subplots_adjust(wspace=00.5)
plt.show()
