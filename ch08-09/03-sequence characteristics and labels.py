import random
import torch
from d2l import torch as d2l


# 随机生成一个小批量数据的特征和标签以供读取
def seq_data_iter_random(corpus, batch_size, num_steps):  # numsteps相当于马尔可夫的τ
    # 随机指定开始的index（0~τ-1），后面划分以此开始，τ个为间隔划分。包含所有的划分可能性
    corpus = corpus[random.randint(0, num_steps-1):]     # 左闭右闭
    num_subseqs = (len(corpus)-1)//num_steps  # 子序列的个数，//:向下取整 减去1：最后一个数没有后面的真实标签要减去
    initial_indices = list(range(0, num_subseqs*num_steps, num_steps))    # 每个子序列开始的index
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos:pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i:i+batch_size]
        x = [data(j) for j in initial_indices_per_batch]
        y = [data(j+1) for j in initial_indices_per_batch]
        yield torch.tensor(x), torch.tensor(y)


my_seq = list(range(35))
for x, y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('x:', x, '\ny:', y)


# # 随机生成一个小批量数据的特征和标签以供读取，与上不同的是相邻两个样本在原始序列相邻
def seq_data_iter_sequential(corpus, batch_size, num_steps):  # numsteps相当于马尔可夫的τ
    # 随机指定开始的index（0~τ-1），后面划分以此开始，τ个为间隔划分。包含所有的划分可能性
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus)-offset-1)//batch_size)*batch_size
    xs = torch.tensor(corpus[offset:offset+num_tokens])
    ys = torch.tensor(corpus[offset+1:offset+1+num_tokens])
    xs, ys = xs.reshape(batch_size, -1), ys.reshape(batch_size, -1)
    num_batches = xs.shape[1]//num_steps
    for i in range(0, num_steps*num_batches, num_steps):
        x = xs[:, i:i+num_steps]
        y = ys[:, i:i+num_steps]
    yield x, y


for x, y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('x:', x, '\ny:', y)


class SeqDataLoder:
    def __init__(self, batch_size, num_steps, use_randon_iter, max_tokens):
        if use_randon_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)  # 01里面的最后一个函数
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

# 以上是d2l中一些函数包


# 以下专门针对时间机器数据集
def load_data_time_machine(batch_size, num_steps, use_random_iter, max_tokens):
    data_iter = SeqDataLoder(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
