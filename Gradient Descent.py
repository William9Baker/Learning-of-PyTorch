#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 15:34
# @Author  : William Baker
# @FileName: gd.py
# @Software: PyCharm
# @Blog    : https://blog.csdn.net/weixin_43051346

# Gradent Descent 梯度下降
import matplotlib.pyplot as plt

# 准备数据集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 初始化权重值 w
w = 1.0
# w = 5.0

# 学习率 learning rate
lr = 0.01

# 定义模型-线性模型 y=w*x
def forward(x):
    return w*x

# 定义代价函数cost
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):    # zip参考： https://www.cnblogs.com/shixiaoxun/articles/14200157.html
        y_pred = forward(x)     # 预测值 y_hat
        cost += (y_pred - y) ** 2   # MSE 均方误差
    return cost/len(xs)

# 定义梯度
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        grad += (y_pred - y) * 2 * x     # changed here
    return grad/len(xs)


epoch_list = []
cost_list = []
print('训练前的输入值x：{}, 训练前的预测值：{}\n'.format(4.0, forward(4.0)))
print("***************************开始训练***************************")
# 开始训练
for epoch in range(100):
    cost_val = cost(x_data, y_data)   # 预测的loss值
    grad_val = gradient(x_data, y_data)   # 预测的gradient
    w -= lr * grad_val     # w = w - lr * gradient(w)   梯度下降的核心所在
    # w -= 0.1 * w
    # print('Epoch:{}, w={}, loss={}'.format(epoch, w, cost_val))
    print('Epoch:{}, w={}, loss={}, grad={}'.format(epoch, w, cost_val, grad_val))

    epoch_list.append(epoch)
    cost_list.append(cost_val)

print("***************************训练结束***************************\n")
print('训练后的输入值x：{}, 训练后的预测值：{}'.format(4.0, forward(4)))

# 绘图
plt.plot(epoch_list, cost_list)
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()
plt.savefig("./gd.png")