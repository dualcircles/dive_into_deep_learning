from d2l import torch as d2l

tokens = d2l.tokenize(d2l.read_time_machine())
corpus = [token for line in tokens for token in line]

# 一元语法
vocab = d2l.Vocab(corpus)
print(vocab.token_freqs[:10])
freqs = [freq for token, freq in vocab.token_freqs]
# log运算后，轴的取值就会以一个量级增长。例如，以10的指数增长
d2l.plot(freqs,  xlabel='token:x', ylabel='frenquency:n(x)', xscale='log', yscale='log')
# 最流行的词被称为停用词
# d2l.plt.show()

# 二元语法
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])

# 三元语法
trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs],  xlabel='token:x', ylabel='frenquency:n(x)',
         xscale='log', yscale='log', legend=['unigram', 'bigram', 'trigram'])
# 最流行的词被称为停用词
d2l.plt.show()


