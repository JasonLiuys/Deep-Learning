import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y ,t)

        return loss

net = simpleNet()
print(net.W)

x = np.array([0.6,0.9])
p = net.predict(x)
print(p)

maxp = np.argmax(p)
print(maxp)

t = np.array([0,0,1])
net.loss(x,t)
#以上实现了一个简单的神经网络

#伪函数F(W)
def f(W):
    return net.loss(x,t)

#计算梯度
dW = numerical_gradient(f, net.W)
print(dW)

