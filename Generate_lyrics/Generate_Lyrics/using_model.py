import tensorflow as tf
import generate_lyrics

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