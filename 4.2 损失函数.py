
# coding: utf-8

# In[6]:


import numpy as np 
def mean_square_error(y,t):
    return 0.5 * np.sum((y-t)**2)
#均方误差 y与t为numpy数组

def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t*np.log(y + delta))
#交叉熵误差

t = [0,0,1,0,0,0,0,0,0,0]
y1 = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
y2 = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
mean_square_error(np.array(y1),np.array(t))
cross_entropy_error(np.array(y2),np.array(t))


# In[13]:


#mini-batch
import sys,os
sys.path.append(os.pardir)
import numpy as np 
from dataset.mnist import load_mnist

(x_train , t_train) , (x_test , t_test) = load_mnist(normalize = True , one_hot_label = True)

print (x_train.shape) #60000个数据 28*28 = 784维
print (t_train.shape) #60000个数据 10维（0-9）

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size , batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(batch_mask)


# In[ ]:


#mini_batch的交叉熵误差（one-hot表示）
def cross_entropy_error(y ,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
        
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7))/batch_size


# In[ ]:


#mini_batch的交叉熵误差（非one-hot表示）
def cross_entropy_error(y ,t):
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
        
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size

