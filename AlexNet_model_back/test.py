import os  # 导入os模块，用于与操作系统交互
import sys  # 导入sys模块，用于操作Python运行时环境

sys.path.append(os.getcwd())  # 将当前工作目录添加到sys.path，方便模块导入
import torch  # 导入PyTorch主库
from torch.utils.data import (
    DataLoader,  # 导入DataLoader用于批量加载数据
    random_split,  # 导入random_split用于划分数据集（本文件未用到）
)
from torchvision import datasets, transforms  # 导入torchvision的数据集和数据变换模块
from torchvision.datasets import FashionMNIST  # 导入FashionMNIST数据集
from AlexNet_model_back.model import AlexNet  # 从自定义模块导入AlexNet模型


def test_data_load():  # 定义测试数据加载函数
    test_dataset = FashionMNIST(
        root="./data",  # 数据存储路径
        train=False,  # 加载测试集
        download=True,  # 如果数据不存在则下载
        transform=transforms.Compose(
            [
                transforms.Resize(size=227),  # 将图片缩放到227x227
                transforms.ToTensor(),  # 转换为Tensor
            ]
        ),
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=1,  # 测试集加载器，批量128，打乱顺序
    )

    return test_loader  # 返回测试集加载器


print(test_data_load())  # 打印测试集加载器（调试用）


def test_model_process(model, test_loader):  # 定义模型测试过程
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 判断是否有GPU可用
    model.to(device)  # 将模型移动到指定设备
    model.eval()  # 设置模型为评估模式

    correct = 0  # 正确预测样本数
    total = 0  # 总样本数

    with torch.no_grad():  # 关闭梯度计算，加快推理速度
        for images, labels in test_loader:  # 遍历测试集
            images, labels = images.to(device), labels.to(device)  # 数据移动到设备
            outputs = model(images)  # 前向传播，获取输出
            _, predicted = torch.max(outputs, 1)  # 获取预测标签
            total += labels.size(0)  # 累加总样本数
            correct += torch.sum(predicted == labels.data)  # 累加正确预测数

    accuracy = correct / total * 100  # 计算准确率（百分比）
    print(f"Test Accuracy: {accuracy:.2f}%")  # 打印测试准确率


if __name__ == "__main__":  # 如果当前脚本作为主程序运行
    test_loader = test_data_load()  # 加载测试集
    model = AlexNet()  # 实例化AlexNet模型
    model.load_state_dict(
        torch.load("./models/alex_net_best_model.pth")
    )  # 加载训练好的模型参数
    test_model_process(model, test_loader)  # 测试模型并输出准确率
