import os  # 导入os模块，用于与操作系统交互
import sys  # 导入sys模块，用于操作Python运行时环境

sys.path.append(os.getcwd())  # 将当前工作目录添加到sys.path，方便模块导入


from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from ResNet_model_cat_dog.model import ResNet


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


model = ResNet()
model.load_state_dict(torch.load("./models/cat_dog_res_net_best_model.pth"))
device = "cuda" if torch.cuda.is_available() else "cpu"  # 判断是否有GPU可用
model.to(device)  # 将模型移动到指定设备
model.eval()  # 设置模型为评估模式

# 遍历加载data\dogs-vs-cats-redux-kernels-edition\test\test 目录下.jpg 文件

dir = "./data/dogs-vs-cats-redux-kernels-edition/test/test"
for root, dirs, files in os.walk(dir):
    for file in files:
        if file.endswith(".jpg"):
            # print(f"Found image: {file}")  # 打印找到的图片文件名

            image = Image.open(os.path.join(dir, file)).convert("RGB")  # 打开测试图片
            image_tensor = train_transform(image)  # 应用预处理转换
            image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度，变为

            # print(image_tensor.shape)  # 打印张量形状，应该是(1, 3, 227, 227)

            with torch.no_grad():  # 关闭梯度计算
                image_tensor = image_tensor.to(device)  # 将图像张量移动到设备
                output = model(image_tensor)  # 前向传播，获取输出
                _, predicted = torch.max(output, 1)  # 获取预测标签
                result = predicted.item()  # 获取预测结果的整数值

            classes = ["cat", "dog"]  # 定义类别列表
            print(f"{file} 预测结果：{classes[result]}")  # 打印预测结果
