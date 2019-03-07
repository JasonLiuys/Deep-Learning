#对歌词进行分词
#coding : utf8

import jieba

def preprocess(data):
    data = data.replace('《', ' ')
    data = data.replace('》', ' ')
    data = data.replace('【', ' ')
    data = data.replace('】', ' ')
    data = data.replace(' ', ';')
    data = data.replace('\n', '.')

    words = jieba.lcut(data, cut_all = False)

    return words

def write_file(words , fname):
    with open(fname , 'a' , encoding='utf-8') as f :
        for w in words:
            f.write(w + '\n')

    print('Done')

if __name__ == "__main__":
    with open("data/lyrics.txt" , encoding='utf-8') as f:
        text = f.read()

    words = preprocess(text)
    write_file(words , "data/split.txt")