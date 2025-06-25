import os  # 导入os模块，用于与操作系统交互
import sys  # 导入sys模块，用于操作Python运行时环境

sys.path.append(os.getcwd())  # 将当前工作目录添加到sys.path，方便模块导入

import time  # 导入time模块，用于计时
from torchvision.datasets import ImageFolder  # 导入FashionMNIST数据集
from torchvision import transforms  # 导入transforms用于数据预处理
from torch.utils.data import (
    DataLoader,  # 导入DataLoader用于批量加载数据
    random_split,  # 导入random_split用于划分数据集
)
import numpy as np  # 导入numpy用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib用于绘图
import torch  # 导入PyTorch主库
from torch import nn, optim  # 导入神经网络模块和优化器
import copy  # 导入copy模块用于深拷贝
import pandas as pd  # 导入pandas用于数据处理

from GoogLeNet_model_cat_dog.model import GoogLeNet  # 从自定义模块导入GoogLeNet

batch_size = 32  # 定义批量大小


def train_val_date_load():  # 定义函数用于加载训练集和验证集
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),  # 随机水平翻转图片
            transforms.RandomRotation(10),  # 随机旋转图片，角度范围
            transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),  # 随机裁剪图片
            transforms.ColorJitter(  # 随机调整亮度、对比度、饱和度
                brightness=0.2,  # 亮度调整范围
                contrast=0.2,  # 对比度调整范围
                saturation=0.2,  # 饱和度调整范围
            ),
            transforms.Resize(size=(224, 224)),  # 227x227缩放图片
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

    ROOT_TRAIN = r"data/dogs-vs-cats-redux-kernels-edition/train/train"
    train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)

    train_date, val_data = random_split(
        train_dataset,
        [
            int(len(train_dataset) * 0.8),  # 80%作为训练集
            len(train_dataset) - int(len(train_dataset) * 0.8),  # 剩余20%作为验证集
        ],
    )

    train_loader = DataLoader(
        dataset=train_date,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,  # 训练集加载器，批量32，打乱顺序
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,  # 验证集加载器，批量32，打乱顺序
    )

    return train_loader, val_loader  # 返回训练集和验证集加载器


def train_model_process(model, train_loader, val_loader, epochs=10):  # 定义训练过程函数
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 判断是否有GPU可用
    optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 使用Adam优化器，学习率0.001
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    model.to(device)  # 将模型移动到指定设备

    best_model_wts = copy.deepcopy(model.state_dict())  # 保存最佳模型参数
    best_acc = 0.0  # 初始化最佳准确率
    train_loss_all = []  # 记录每轮训练损失
    val_loss_all = []  # 记录每轮验证损失
    train_acc_all = []  # 记录每轮训练准确率
    val_acc_all = []  # 记录每轮验证准确率

    since = time.time()  # 记录训练开始时间

    for epoch in range(epochs):  # 遍历每个训练轮次
        print(f"Epoch {epoch + 1}/{epochs}")  # 打印当前轮次信息

        train_loss = 0.0  # 当前轮训练损失
        train_correct = 0  # 当前轮训练正确样本数

        val_loss = 0.0  # 当前轮验证损失
        val_correct = 0  # 当前轮验证正确样本数

        train_num = 0  # 当前轮训练样本总数
        val_num = 0  # 当前轮验证样本总数

        for step, (images, labels) in enumerate(train_loader):  # 遍历训练集
            images = images.to(device)  # 将图片移动到设备
            labels = labels.to(device)  # 将标签移动到设备

            model.train()  # 设置模型为训练模式

            outputs = model(images)  # 前向传播，获取输出

            pre_lab = torch.argmax(outputs, dim=1)  # 获取预测标签

            loss = criterion(outputs, labels)  # 计算损失

            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            train_loss += loss.item() * images.size(0)  # 累加损失
            train_correct += torch.sum(pre_lab == labels.data)  # 累加正确预测数
            train_num += labels.size(0)  # 累加样本数
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc:{:.4f}".format(
                    epoch + 1,
                    epochs,
                    step + 1,
                    len(train_loader),
                    loss.item(),
                    torch.sum((pre_lab == labels.data) / batch_size),
                )
            )

        for step, (images, labels) in enumerate(val_loader):  # 遍历验证集
            images = images.to(device)  # 将图片移动到设备
            labels = labels.to(device)  # 将标签移动到设备
            model.eval()  # 设置模型为评估模式

            with torch.no_grad():  # 关闭梯度计算
                outputs = model(images)  # 前向传播
                pre_lab = torch.argmax(outputs, dim=1)  # 获取预测标签
                loss = criterion(outputs, labels)  # 计算损失

                val_loss += loss.item() * images.size(0)  # 累加损失
                val_correct += torch.sum(pre_lab == labels.data)  # 累加正确预测数
                val_num += labels.size(0)  # 累加样本数
                print(
                    "Epoch [{}/{}], Step [{}/{}], Val Loss: {:.4f}, Acc:{:.4f}".format(
                        epoch + 1,
                        epochs,
                        step + 1,
                        len(val_loader),
                        loss.item(),
                        torch.sum((pre_lab == labels.data) / batch_size),
                    )
                )

        train_loss_all.append(train_loss / train_num)  # 记录本轮平均训练损失
        val_loss_all.append(val_loss / val_num)  # 记录本轮平均验证损失
        train_acc = train_correct.double() / train_num  # 计算本轮训练准确率
        val_acc = val_correct.double() / val_num  # 计算本轮验证准确率
        train_acc_all.append(train_acc.item())  # 记录训练准确率
        val_acc_all.append(val_acc.item())  # 记录验证准确率
        print(
            f"Train Loss: {train_loss / train_num:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss / val_num:.4f}, Val Acc: {val_acc:.4f}"
        )  # 打印本轮损失和准确率
        if val_acc_all[-1] > best_acc:  # 如果本轮验证准确率更高
            best_acc = val_acc_all[-1]  # 更新最佳准确率
            best_model_wts = copy.deepcopy(model.state_dict())  # 保存最佳模型参数

        # model.load_state_dict(best_model_wts)  # 可选：恢复最佳模型参数

    time_elapsed = time.time() - since  # 计算训练总耗时
    print(
        f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n"
        f"Best val Acc: {best_acc:.4f}"
    )  # 打印训练耗时和最佳准确率

    torch.save(
        model.state_dict(), "./models/cat_dog_google_net_best_model.pth"
    )  # 保存模型参数
    train_process = pd.DataFrame(
        data={
            "epoch": range(1, epochs + 1),  # 轮次
            "train_loss_all": train_loss_all,  # 训练损失
            "val_loss_all": val_loss_all,  # 验证损失
            "train_acc_all": train_acc_all,  # 训练准确率
            "val_acc_all": val_acc_all,  # 验证准确率
        }
    )  # 构建训练过程数据表

    return train_process  # 返回训练过程数据


def matplot_acc_loss(train_process):  # 定义绘图函数
    plt.figure(figsize=(12, 5))  # 创建画布，设置大小

    plt.subplot(1, 2, 1)  # 激活第1个子图
    plt.plot(
        train_process["epoch"], train_process["train_loss_all"], label="Train Loss"
    )  # 绘制训练损失曲线
    plt.plot(
        train_process["epoch"], train_process["val_loss_all"], label="Val Loss"
    )  # 绘制验证损失曲线
    plt.xlabel("Epoch")  # 设置x轴标签
    plt.ylabel("Loss")  # 设置y轴标签
    plt.title("Loss vs Epoch")  # 设置子图标题
    plt.legend()  # 显示图例

    plt.subplot(1, 2, 2)  # 激活第2个子图
    plt.plot(
        train_process["epoch"], train_process["train_acc_all"], label="Train Acc"
    )  # 绘制训练准确率曲线
    plt.plot(
        train_process["epoch"], train_process["val_acc_all"], label="Val Acc"
    )  # 绘制验证准确率曲线
    plt.xlabel("Epoch")  # 设置x轴标签为Epoch
    plt.ylabel("Accuracy")  # 设置y轴标签为Accuracy
    plt.title("Accuracy vs Epoch")  # 设置子图标题
    plt.legend()  # 显示图例

    plt.tight_layout()  # 自动调整子图间距
    plt.ion()  # 关闭交互模式，防止图像自动关闭
    plt.show()  # 显示所有图像
    plt.savefig("./models/cat_dog_google_net_output.png")  # 保存图片到指定路径


if __name__ == "__main__":  # 如果当前脚本作为主程序运行
    traindatam, valdata = train_val_date_load()  # 加载训练集和验证集
    result = train_model_process(
        GoogLeNet(2), traindatam, valdata, 10
    )  # 训练模型并获取训练过程数据
    matplot_acc_loss(result)  # 绘制训练和验证的损失及准确率曲线
