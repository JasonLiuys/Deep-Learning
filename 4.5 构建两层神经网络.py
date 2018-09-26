from typing import Union

import numpy as np
import sys, os

from numpy.core.multiarray import ndarray

sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
#激活函数

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
# 交叉熵误差

#实现两层神经网络
class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01): #初始化，输入各层的神经元数
        #初始化权重
        self.params = {} #保存神经网络的参数 w为权重 b为偏置
        self.params['W1']= weight_init_std * np.random.randn(input_size, output_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['W2']=weight_init_std * np.random.randn(input_size, output_size)
        self.params['b2']=np.zeros(hidden_size)

    def predict(self, x):   #进行前向处理，x是图像数据
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x :输入数据， t :监督数据
    def loss(self, x, t): #计算损失函数的值 参数x是图像数据， t是正确解标签
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):  #计算识别精度
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t): #计算权重参数的梯度
        loss_W = lambda W: self.loss(x, t)

        grads = {} #保存梯度
        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) #第一层权重的梯度
        grads['b1'] = numerical_gradient(loss_W, self.params['b1']) #第一层偏置的梯度
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

#实现mini-batch
(x_train, t_train), (x_test, t_test) =  load_mnist(normalize=True, one_hot_label = True)

train_loss_list = []
train_acc_list = []
test_acc_list = []

#超参数
iters_num = 10000
train_size = x_train.shape[0] #60000
batch_size = 100
learning_rate = 0.1

#平均每个epoch的重复次数
iter_per_epoch = max(train_size / batch_size , 1)

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

for i in range(iters_num):
    #获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    print(x_batch)
    t_batch = t_train[batch_mask]
    #print(t_batch)

    #计算梯度
    grad = network.numerical_gradient(x_batch, t_batch)

    #更新参数
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    #记录学习过程
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    #计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
