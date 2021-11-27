#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 17:37
# @Author  : William Baker
# @FileName: SGD_torch.py
# @Software: PyCharm
# @Blog    : https://blog.csdn.net/weixin_43051346

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import matplotlib.pyplot as plt


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

#tensor中包含data(w)和grad(loss对w求导)
w = torch.tensor([1.0])     # w的初值为1.0
w.requires_grad = True      # 需要计算梯度
print(w.data)
# 学习率 learning rate
lr = 0.01

def forward(x):
    return x * w     # x和w都是tensor类型，数乘

def loss(x, y):
    # 每调用一次loss函数，就把计算图构建出来了
    y_pred = forward(x)
    return (y_pred - y) ** 2

# print("predict (before training)", 4, forward(4).item())
print('训练前的输入值x：{}, 训练前的预测值：{}\n'.format(4, forward(4).item()))
print("***************************开始训练***************************")    # 训练当达到最优值时，即损失函数梯度grad下降到梯度为0，W的值不再继续迭代

epoch_list = []
cost_list = []
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        # 下面两行，构建计算图的时候，直接使用张量进行运算（但是在权重更新的时候，要使用.data)
        l = loss(x, y)      # l是一个张量tensor，tensor主要是在建立计算图 forward, compute the loss，Forward 前馈是计算损失loss，创建新的计算图
        l.backward()        # backward,compute grad for Tensor whose requires_grad set to True 反向传播过程就会自动计算所需要的梯度
                            # Backward 反馈是计算梯度，计算完梯度后会将其存到变量（比如权重w）里面，存完之后计算图就会得到释放
                            # 每进行一次反向传播，把图释放，准备进行下一次的图
        """
        # print('\tgrad:', x, y, w.grad.item(), w.grad.item().type)    # 报错 AttributeError: 'float' object has no attribute 'type'
        print('\tgrad:', x, y, w.grad.item(), type(w.grad.item()))     # w.grad.item()   # grad:  2.0 4.0 -7.840000152587891 <class 'float'>
        # print('\tgrad-:', x, y, w.grad, type(w.grad))                # w.grad          # grad-: 2.0 4.0 tensor([-7.8400]) <class 'torch.Tensor'>
        print('\tgrad-*:', x, y, w.grad, w.grad.type())                # w.grad          # grad-*: 2.0 4.0 tensor([-7.8400]) torch.FloatTensor
        print('\tgrad--:', x, y, w.grad.data, w.grad.data.type())      # w.grad.data     # grad--: 2.0 4.0 tensor([-7.8400]) torch.FloatTensor
        """
        print('\tgrad:', x, y, w.grad.item())                           # grad: 2.0 4.0 -7.840000152587891
        # print('\tgrad--:', x, y, w.data.item(), type(w.data.item()))    # grad--: 2.0 4.0 1.0199999809265137 <class 'float'>

        # w -= lr * grad_val        # w = w - lr * gradient(w)   梯度下降的核心所在
        # print(w.data.requires_grad)    # False
        w.data = w.data - lr * w.grad.data       # 权重更新时，需要用到标量，注意grad也是一个tensor   # w.grad.item()是等价于 w.grad.data的，都是不建立计算图
        # print(w.data.requires_grad)    # False

        w.grad.data.zero_()     # after update, remember set the grad to zero     # 把权重里面的梯度数据清0，不然就变成了梯度累加

    epoch_list.append(epoch)
    cost_list.append(l)
    # print('progress:', epoch, l.item())
    print('Progress: Epoch {}, loss:{}'.format(epoch, l.item()))    # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）
                                                                          # Progress: Epoch 99, loss:9.094947017729282e-13
    # print('Progress-: Epoch {}, loss:{}'.format(epoch, l.data.item()))  # Progress-: Epoch 99, loss:9.094947017729282e-13

print("***************************训练结束***************************\n")
# print("predict (after training)", 4, forward(4).item())
print('训练后的输入值x：{}, 训练后的预测值：{}'.format(4, forward(4).item()))

# 绘图
plt.plot(epoch_list, cost_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()