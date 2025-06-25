import os  # 导入os模块，用于处理文件和目录
import sys

sys.path.append(os.getcwd())  # 添加上级目录到系统路径，以便导入其他模块

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from LetNet5_model_cat_dog.model import LeNet5

image = Image.open("cat_test_01.png").convert("RGB")  # 打开测试图片


train_transform = transforms.Compose(
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

image_tensor = train_transform(image)  # 应用预处理转换
image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度，变为


print(image_tensor.shape)  # 打印张量形状，应该是(1, 3, 227, 227)


model = LeNet5(2)
model.load_state_dict(torch.load("./models/cat_dog_le_net5_best_model.pth"))
device = "cuda" if torch.cuda.is_available() else "cpu"  # 判断是否有GPU可用
model.to(device)  # 将模型移动到指定设备
model.eval()  # 设置模型为评估模式


with torch.no_grad():  # 关闭梯度计算
    image_tensor = image_tensor.to(device)  # 将图像张量移动到设备
    output = model(image_tensor)  # 前向传播，获取输出
    _, predicted = torch.max(output, 1)  # 获取预测标签
    result = predicted.item()  # 获取预测结果的整数值

classes = ["cat", "dog"]  # 定义类别列表
print(f"预测结果：{classes[result]}")  # 打印预测结果
