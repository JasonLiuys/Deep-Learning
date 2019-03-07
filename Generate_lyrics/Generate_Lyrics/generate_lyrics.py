#一.读取数据
def load_data(fname):
    with open(fname , 'r', encoding='utf-8') as f:
        text = f.read()

    data = text.split()
    return data

text = load_data('data/split.txt')
print("前十个词：{}".format(text[:10]))

#二.数据预处理
#构造词典及映射
vocab = set(text) #set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据
vocab_to_int = {w : idx for (idx , w) in enumerate(vocab)}  #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
int_to_vocab = {idx : w for (idx , w) in enumerate(vocab)}

print('Total words: {}'.format(len(text)))
print('Vocab size: {}'.format(len(vocab)))

#转换文本为整数
int_text = [vocab_to_int[w] for w in text]
print("int_text:" , int_text)

#三.构建网络
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt

# # Check TensorFlow Version
# assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
# print('TensorFlow版本: {}'.format(tf.__version__))
#
# # Check for a GPU
# if not tf.test.gpu_device_name():
#     warnings.warn('未发现GPU，请使用GPU进行训练！')
# else:
#     print('默认GPU设备: {}'.format(tf.test.gpu_device_name()))

#输入层
def get_input():
    inputs = tf.placeholder(tf.int32 , [None , None] , name = "inputs")
    targets = tf.placeholder(tf.int32 , [None , None] , name = "targets")
    learning_rate = tf.placeholder(tf.float32 , name = "learning_rate")
    return inputs , targets , learning_rate

#RNN cell
def get_init_cell(batch_size , rnn_size):
    #rnn_size为RNN隐层神经元个数
    lstm = rnn.BasicLSTMCell(rnn_size)
    cell = rnn.MultiRNNCell([lstm]) #https://www.leiphone.com/news/201709/QJAIUzp0LAgkF45J.html

    initial_state = cell.zero_state(batch_size , tf.float32) #返回[batch_size, len(cells)]这个函数只是用来生成初始化值的
    initial_state = tf.identity(initial_state , "initial_state") #它返回一个和输入的 tensor 大小和数值都一样的 tensor ,类似于 y=x 操作
    return cell , initial_state

#word_embedding
def get_embed(input_data , vocab_size , embed_dim):
    #单词太多需要embedding
    # input_data: 输入的tensor
    # vocab_size: 词汇表大小
    # embed_dim: 嵌入维度
    embedding = tf.Variable(tf.random_uniform([vocab_size, embed_dim], -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data) #embedding_lookup（params, ids）函数的用法主要是选取一个张量里面索引对应的元素。params可以是张量也可以是数组等，id就是对应的索引。

    return embed

#Build RNN
def build_rnn(cell , inputs):
    #构建RNN模型
    # cell: RNN单元
    # inputs: 输入的batch
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32) #output与h其实为一样的，需要将output进行其他操作才能得到真正输出

    final_state = tf.identity(final_state, 'final_state')
    return outputs, final_state

#Build Neural Network
def build_nn(cell , rnn_size , input_data , vocab_size , embed_dim):
    '''
       构建神经网络，将RNN层与全连接层相连

       参数:
       ---
       cell: RNN单元
       rnn_size: RNN隐层结点数量
       input_data: input tensor
       vocab_size
       embed_dim: 嵌入层大小
       '''
    embed = get_embed(input_data , vocab_size , embed_dim)
    outputs , final_state = build_rnn(cell , embed)

    logits = tf.contrib.layers.fully_connected(outputs , vocab_size , activation_fn = None)

    return logits , final_state

#构造batch
    #
    # 构造batch
    # 在这里，我们将采用以下方式进行batch的构造，如果我们有一个1-20的序列，传入参数batch_size=3, seq_length=2的话，希望返回以下一个四维的向量。
    #
    # 分为了三个batch，每个batch中包含了输入和对应的目标输出。 get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2)
    #
    # First Batch
    # [
    #
    # Batch of Input
    # [[ 1  2], [ 7  8], [13 14]]
    # # Batch of targets
    # [[ 2  3], [ 8  9], [14 15]]
    # ]
    #
    # Second Batch
    # [
    #
    # # Batch of Input
    # [[ 3  4], [ 9 10], [15 16]]
    # # Batch of targets
    # [[ 4  5], [10 11], [16 17]]
    # ]
    #
    # Third Batch
    # [
    #
    # # Batch of Input
    # [[ 5  6], [11 12], [17 18]]
    # # Batch of targets
    # [[ 6  7], [12 13], [18  1]]
    # ] ]
def get_batches(int_text, batch_size, seq_length):
    '''
    构造batch
    '''
    batch = batch_size * seq_length
    n_batch = len(int_text) // batch

    int_text = np.array(int_text[:batch * n_batch])  # 保留能构成完整batch的数量

    int_text_targets = np.zeros_like(int_text)
    int_text_targets[:-1], int_text_targets[-1] = int_text[1:], int_text[0]

    # 切分
    x = np.split(int_text.reshape(batch_size, -1), n_batch, -1)
    y = np.split(int_text_targets.reshape(batch_size, -1), n_batch, -1)

    return np.stack((x, y), axis=1)  # 组合

#四.模型训练
num_epochs = 100
batch_size = 64
rnn_size = 512
embed_dim = 200 #Embedding Dimension Size
seq_length = 20
learning_rate = 0.001
show_every_n_batches = 100
loss_array = []

from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    vocab_size = len(int_to_vocab)
    input_text , targets , lr = get_input() #输入tensor
    input_data_shape = tf.shape(input_text)

    #初始化RNN
    cell , initial_state = get_init_cell(input_data_shape[0] , rnn_size) #input_data_shape[0]为mini_batch的size
    logits , final_state = build_nn(cell , rnn_size , input_text , vocab_size , embed_dim)

    #计算softmax层概率
    probs = tf.nn.softmax(logits , name = "probs")

    #损失函数
    cost = seq2seq.sequence_loss(logits , targets , tf.ones([input_data_shape[0] , input_data_shape[1]]))

    #优化函数
    optimizer = tf.train.AdamOptimizer(lr)

    #梯度裁剪
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad , -1. , 1.), var) for grad , var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

    #获取batch
    batches = get_batches(int_text , batch_size , seq_length)
    save_dir = './save'

    with tf.Session(graph = train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            state = sess.run(initial_state , {input_text : batches[0][0]})

            for batch_i , (x , y) in enumerate(batches):
                feed = {
                    input_text : x ,
                    targets : y ,
                    initial_state : state ,
                    lr : learning_rate
                }
                train_loss , state , _ = sess.run([cost , final_state , train_op] , feed)
                loss_array.append(train_loss)

                # 每训练一定阶段对结果进行打印
                if (epoch * len(batches) + batch_i) % show_every_n_batches == 0:
                    print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        epoch,
                        batch_i,
                        len(batches),
                        train_loss))

        #绘制损失图
        x_axis = range(len(loss_array))
        plt.plot(x_axis, loss_array)
        plt.title('loss for each batch')
        plt.show()
        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, save_dir)
        print('Model Trained and Saved')

#五.使用模型
def get_tensors(loaded_graph):
    '''
    获取模型训练结果参数
    参数
    ---
    loaded_graph: 从文件加载的tensroflow graph
    '''
    inputs = loaded_graph.get_tensor_by_name('inputs:0')
    initial_state = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state = loaded_graph.get_tensor_by_name('final_state:0')
    probs = loaded_graph.get_tensor_by_name('probs:0')
    return inputs, initial_state, final_state, probs


def pick_word(probabilities, int_to_vocab):
    '''
    选择单词进行文本生成，用来以一定的概率生成下一个词
    参数
    ---
    probabilities: Probabilities of the next word
    int_to_vocab: 映射表
    '''
    result = np.random.choice(len(probabilities), 50, p=probabilities) #根据probably从所有里选50个
    return int_to_vocab[result[0]]


# 生成文本的长度
gen_length = 300

# 定义冷启动的单词
prime_word = '离开'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # 加载模型
    loader = tf.train.import_meta_graph(save_dir + '.meta')
    loader.restore(sess, save_dir)

    # 获取训练的结果参数
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    # Sentences generation setup
    gen_sentences = [prime_word]
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    # 生成句子
    for n in range(gen_length):
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0]) # =1

        # 预测
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})

        pred_word = pick_word(probabilities[dyn_seq_length - 1], int_to_vocab)

        gen_sentences.append(pred_word)

    lyrics = ' '.join(gen_sentences)
    lyrics = lyrics.replace(';', '\n')
    lyrics = lyrics.replace('.', ' ')
    lyrics = lyrics.replace(' ', '')

    print(lyrics)





