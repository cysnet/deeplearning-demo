import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np


def compute_normalization_params(dataset_path, batch_size=32, num_workers=4):
    """
    计算自定义数据集的均值和标准差

    参数:
        dataset_path: 数据集路径
        batch_size: 批量大小
        num_workers: 数据加载工作线程数

    返回:
        mean: 各通道均值 [R, G, B]
        std: 各通道标准差 [R, G, B]
    """
    # 创建无归一化的transform
    transform = transforms.Compose(
        [
            transforms.Resize((227, 227)),  # 统一大小
            transforms.ToTensor(),  # 转换为Tensor
        ]
    )

    # 加载数据集
    dataset = ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    # 初始化变量
    mean = 0.0
    std = 0.0
    nb_samples = 0.0

    # 遍历数据集计算均值和方差
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    # 计算整体均值和标准差
    mean /= nb_samples
    std /= nb_samples

    return mean.tolist(), std.tolist()


# 使用示例
if __name__ == "__main__":
    dataset_path = "data/dogs-vs-cats-redux-kernels-edition/train/train"
    mean, std = compute_normalization_params(dataset_path)
    print(mean, std)
    print(f"计算得到的均值: {mean}")
    print(f"计算得到的标准差: {std}")
