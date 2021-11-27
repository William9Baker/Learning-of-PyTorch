#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 16:52
# @Author  : William Baker
# @FileName: SGD.py
# @Software: PyCharm
# @Blog    : https://blog.csdn.net/weixin_43051346


# SGD 随机梯度下降
import matplotlib.pyplot as plt

# 准备数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始化权重值 w
w = 1.0

# 学习率 learning rate
lr = 0.01

# 定义模型-线性模型 y=w*x
def forward(x):
    return w*x

# 定义损失函数loss
def loss(x, y):
    y_pred = forward(x)     # 预测值 y_hat
    return (y_pred - y) ** 2

# 定义梯度
def gradient(x, y):
    y_pred = forward(x)
    return 2 * x * (y_pred - y)


epoch_list = []
cost_list = []
print('训练前的输入值x：{}, 训练前的预测值：{}\n'.format(4.0, forward(4.0)))
print("***************************开始训练***************************")

# 开始训练
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        loss_val = loss(x, y)   # 预测的loss值
        grad_val = gradient(x, y)   # 预测的gradient
        w -= lr * grad_val     # w = w - lr * gradient(w)   梯度下降的核心所在
    print('Epoch:{}, w={}, loss={}, grad={}'.format(epoch, w, loss_val, grad_val))

    epoch_list.append(epoch)
    cost_list.append(loss_val)

"""
# 开始训练
for epoch in range(100):
    loss_val = loss(x_data, y_data)   # 预测的loss值
    grad_val = gradient(x_data, y_data)   # 预测的gradient
    w -= lr * grad_val     # w = w - lr * gradient(w)   梯度下降的核心所在
    print('Epoch:{}, w={}, loss={}, grad={}'.format(epoch, w, loss_val, grad_val))

    epoch_list.append(epoch)
    cost_list.append(loss_val)
"""

print("***************************训练结束***************************\n")
print('训练后的输入值x：{}, 训练后的预测值：{}'.format(4.0, forward(4)))

# 绘图
plt.plot(epoch_list, cost_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()