import os  # 导入os模块，用于处理文件和目录
import sys

sys.path.append(os.getcwd())  # 添加上级目录到系统路径，以便导入其他模块


import torch
from torch.utils.data import (
    DataLoader,
    random_split,
)
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from LetNet5_model_cat_dog.model import LeNet5


def test_data_load():
    ROOT_TRAIN = r"data/dogs-vs-cats-redux-kernels-edition/test/test"
    test_transform = transforms.Compose(
        [
            transforms.Resize(size=(28, 28)),  # 227x227缩放图片
            transforms.ToTensor(),  # 转换为Tensor格式
            transforms.Normalize(  # 归一化处理
                mean=[
                    0.48832768201828003,
                    0.4550870358943939,
                    0.41696053743362427,
                ],  # RGB通道均值
                std=[
                    0.22568060457706451,
                    0.2211381196975708,
                    0.22132441401481628,
                ],  # RGB通道标准差
            ),
        ]
    )

    test_dataset = ImageFolder(ROOT_TRAIN, transform=test_transform)

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1, shuffle=True, num_workers=1
    )

    return test_loader


print(test_data_load())


def test_model_process(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()  # 设置模型为评估模式

    correct = 0
    total = 0

    classes = ["cat", "dog"]  # 定义类别标签
    with torch.no_grad():  # 在测试时不需要计算梯度
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 前向传播
            _, predicted = torch.max(outputs, 1)  # 获取预测结果
            total += labels.size(0)  # 累计总样本数
            correct += torch.sum(predicted == labels.data)  # 累计正确预测的样本数
            result = predicted.item()  # 获取单个预测结果（调试用）
            label = labels.item()
            print(
                f"Predicted: {result} {classes[result]}, Actual: {label} {classes[label]}"
            )  # 打印预测和实际标签

    accuracy = correct / total * 100  # 计算准确率
    print(f"Test Accuracy: {accuracy:.2f}%")  # 打印测试准确率


if __name__ == "__main__":
    test_loader = test_data_load()  # 加载测试数据
    model = LeNet5(2)  # 实例化LeNet5模型
    model.load_state_dict(
        torch.load("./models/cat_dog_le_net5_best_model.pth")
    )  # 加载模型参数
    test_model_process(model, test_loader)  # 进行模型测试
