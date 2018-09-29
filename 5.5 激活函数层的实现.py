import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self,x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = 1/(1+np.exp(-x))
        self.out = out

        return out

    def backward(self,dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None #损失
        self.y = None #Softmaxd的输出
        self.t = None #监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error() #交叉熵误差、

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)/batch_size

        return dx