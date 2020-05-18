import pickle

import jieba
import numpy as np

with open('data.pk', 'rb') as f:
    all_dick, idf_dict = pickle.load(f)

print(all_dick)
print(idf_dict)


# 按行读取文本文件。
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        fina_outlist = [line.strip() for line in f.readlines()]
    return fina_outlist


# 按行读取词袋文件。每一行按空格切分为一个list，组成2维列表。
def read_file2matrix(file_path):
    fina_outlist = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f.readlines():
            outlist = [float(i) for i in line.strip().split(' ') if i != ' ']
            fina_outlist.append(outlist)
    return fina_outlist


jieba.load_userdict("userdict1.txt")

# 将停用词读出放在stopwords这个列表中
filepath = r'stopwords1.txt'
stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]


def split_words(words):
    word_list = jieba.cut_for_search(words.lower().strip(), HMM=True)
    word_list = [i for i in word_list if i not in stopwords and i != ' ']
    return word_list


# 统计词频，并返回字典
def make_word_freq(word_list):
    freword = {}
    for i in word_list:
        if str(i) in freword:
            freword[str(i)] += 1
        else:
            freword[str(i)] = 1
    return freword


# 计算tfidf,组成tfidf矩阵
def make_tfidf(word_list, all_dick, idf_dict):
    length = len(word_list)
    word_list = [word for word in word_list if word in all_dick]
    word_freq = make_word_freq(word_list)
    w_dic = np.zeros(len(all_dick))
    for word in word_list:
        ind = all_dick[word]
        idf = idf_dict[word]
        w_dic[ind] = float(word_freq[word] / length) * float(idf)
    return w_dic


# 基于numpy的余弦相似性计算
def Cos_Distance(vector1, vector2):
    vec1 = np.array(vector1)
    vec2 = np.array(vector2)
    return float(np.sum(vec1 * vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# 计算相似度
def similarity_words(vec, vecs_list):
    Similarity_list = []
    for vec_i in vecs_list:
        Similarity = Cos_Distance(vec, vec_i)
        Similarity_list.append(Similarity)
    print(np.array(Similarity_list).shape, len(Similarity_list))
    return Similarity_list


def main(words, file_path, readed_path):
    words_list = read_file(file_path)
    # 按行读取文本
    # ['Apple iPhone 8 Plus (A1864) 64GB 深空灰色 移动联通电信4G手机', '荣耀 畅玩7X 4GB+32GB 全网通4G全面屏手机 标配版 铂光金', 'Apple iPhone 8 (A1863) 64GB 深空灰色 移动联通电信4G手机', 'Apple iPhone 7 Plus (A1661) 128G 黑色 移动联通电信4G手机',

    vecs_list = read_file2matrix(readed_path)
    # 按行读取tf-idf词袋
    # [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.791469566521, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    word_list = split_words(words)
    # 待对比语句读取并放在list里
    # ['apple', 'iphone', 'plus', 'a1864', '64gb', '深空', '灰色', '联通', '电信', '4g']

    vec = make_tfidf(word_list, all_dick, idf_dict)
    # 计算tfidf,组成tfidf矩阵
    """
      0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.          0.          0.          0.          0.
  0.          0.          0.79146957  0. 

    """

    similarity_lists = similarity_words(vec, vecs_list)
    # 计算相似度
    # [1.0000000000000002, 0.16116438715917517, 0.85811543720398442, 0.66742265521921251, 0.39477066819942941, 0.60302174693627475, 0.81818098341043122, 0.43081962853380523, 0.35533402523151619, 0.17407746930651016]

    sorted_res = sorted(enumerate(similarity_lists), key=lambda x: x[1])
    # 从小到大排序
    # [(84, 0.0), (87, 0.0), (92, 0.0), (119, 0.0), (134, 0.0), (138, 0.0), (162, 0.0), (294, 0.0), (431, 0.0), (579, 0.0)]

    outputs = [[words_list[i[0]], i[1]] for i in sorted_res[-10:]]
    # 按刚才的顺序取回最后10个句子和相似度
    # [['Apple iPhone 7 Plus (A1661) 128G 黑色 移动联通电信4G手机', 0.66742265521921251]

    return outputs


# words = '回收站'
words = '荣耀 畅玩7X 4GB+32GB 全网通4G全面屏手机 标配版 铂光金'
# words = 'Apple iPhone 8 Plus (A1864) 64GB 深空灰色 移动联通电信4G手机'
# words = '小米8'
# words = "黑色手机"
# words = 'Apple iPhone 8'
# words = '索尼 sony'
file_path = r'MobilePhoneTitle.txt'  # 已经分词的文本
readed_path = r"MobilePhoneTitle_tfidf.txt"  # 已经分词的文本转成tiidf的词袋
outputs = main(words, file_path, readed_path)
# print(outputs)
for i in outputs[::-1]:  # 将句子逆序并打印出来
    print(i[0] + '     ' + str(i[1]))
