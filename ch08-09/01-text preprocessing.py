# 文本预处理，将其当作时序序列,即解决如何将词转换为可以训练的东西
import collections
import re
from d2l import torch as d2l


# 英文数据集, 手动下载了,或者使用02中d2l的方法下载
# d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'time_machine.txt ', '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    with open('time_machine.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    # 将不是大小写的英文字母，全部变成空格，strip()将每一行的回车去掉，再全部变成小写


lines = read_time_machine()
print(lines[0])
print(lines[10])


# 每个文本序列又被拆分成一个标记词表（词元）
def tokenize(lines, token='word'):
    if token == 'word':  # 将序列按单词拆分
        return [line.split() for line in lines]
    elif token == 'char':  # 将序列按字符拆分
        return [list(line) for line in lines]
    else:
        print('错误：未知令牌类型：' + token)


tokens = tokenize(lines)
print(tokens[0])
print(tokens[10])


# 将token映射到从0开始表示的数字索引  （构建一个词典，通常也叫词汇表vocabulary）
class Vocab:
    # min_freq=n 如果一个token出现次数少于n次，就将其丢弃；
    # reserved_token 标记句子的开始和结束token。（这里暂时无用处）
    def __init__(self, tokens=None, min_freq=0, reserved_token=None):
        if tokens is None:
            tokens = []
        if reserved_token is None:
            reserved_token = []
        counter = count_corpus(tokens)  # count_corpus 计数token，函数在下方
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)  # x[1]对第二个元素进行排序从大到小
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_token  # '<unk>'一些特殊的token，且其下标设置为0
        uniq_tokens += [
            token for token, freq in self.token_freqs if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    # 凡是在类中定义了这个__getitem__ 方法，那么它的实例对象（假定为p），可以像这样p[key] 取值，
    # 当实例对象做p[key] 运算时，会调用类中的方法__getitem__

    # 对于给定的一些token返回index
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
            # dict.get(key, default=None) key：要查找的键。default ：如果指定键的值不存在时，返回该默认值。
        return [self.__getitem__(token) for token in tokens]
        #  如果是一些token，将一个一个执行__getitem__函数，就会进入if里面,输出index

    # 对于给定的一些index返回token
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_tokens[index] for index in indices]


# 统计频率,返回字典
def count_corpus(tokens):
    # 这里的tokens是1D列表获知2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):  # 判断两个类型是否相同
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)  # Counter计数里面每一个独一无二token的次数


vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
# 将每一条文本行转换成一个数字索引列表
for i in [0, 10]:
    print('words', tokens[i])
    print('indices:', vocab[tokens[i]])  # 调用了__getitem__


# 打包所有功能
# 返回数据集的标记索引列表和词汇表
def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'word')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# corpus是输入文本数据集转换成的索引数据集，vocab就包括token和index的转换功能
corpus, vocab = load_corpus_time_machine()
print(len(corpus), len(vocab))  # len(vocab) 是字典的大小 调用了__len__
