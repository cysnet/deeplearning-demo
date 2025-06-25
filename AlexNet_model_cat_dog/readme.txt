！！！不收敛的优化


1.降低学习率
optimizer = optim.Adam(model.parameters(), lr=1e-4) 

2.减少 dropout
x = self.fc1(x)  # 第一个全连接层
x = self.ReLU(x)  # 激活
x = F.dropout(x, p=0.2)  # dropout防止过拟合，丢弃概率0.5

x = self.fc2(x)  # 第二个全连接层
x = self.ReLU(x)  # 激活
x = F.dropout(x, p=0.2)  # dropout防止过拟合，丢弃概率0.5