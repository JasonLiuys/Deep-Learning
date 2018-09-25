
# coding: utf-8

# In[22]:


#数值微分：利用微小的查分求导数的过程
#利用中心差分求导数
def numerical_diff(f,x):
    h = 1e-4 #0.0001
    return (f(x+h) - f(x-h))/(2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

import numpy as np 
import matplotlib.pylab as plt
#画出函数1图像
x = np.arange(0.0, 20.0 , 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()

#计算在5与10处的导数
numerical_diff(function_1 , 5)
numerical_diff(function_1 , 10)

def function_2(x):
    return x[0]**2 + x[1]**2

#计算函数2偏导数,重新定义只含一个变量的函数
def function_tmp1(x0):
    return x0*x0 + 4.0**2.0
numerical_diff(function_tmp1 , 3.0)

#梯度:梯度指示的方向是各处函数值减小最多的方向
def numerical_gradient(f,x):
    h = 1e-4
    grad = np.zeros_like(x) #生成与x形状相同的数组
    
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        #f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2)/(2*h)
        x[idx] = tmp_val #还原值
        
    return grad

numerical_gradient(function_2,np.array([0.0,2.0]))

#梯度下降法
def gradient_descent(f, init_x, lr=0.01, step_num=100): #f为进行优化的函数，init_x为初始值，lr为学习次数，step_num为重复次数
    x = init_x
    
    for i in range(step_num):
        grad = numerical_gradient(f ,x) #求梯度函数
        x -= lr*grad
        
    return x

init_x = np.array([-3.0,4.0])
gradient_descent(function_2 , init_x=init_x, lr=0.1, step_num=100 )
    

