import torch
import torch.nn as nn
import time
import math
import random
import gzip
import matplotlib.pyplot as plt
from torch import optim
import torch.nn.functional as F



MAX_LENGTH =200

plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

#
# read file
#
print('reading files...')
file_path = 'acl_titles_and_abstracts.txt'
# file_path = 'acl_titles_and_abstracts.sample.txt'
file = open(file_path).read().split('\n\n')
src_vocab = set()
trg_vocab = set()
for pair in file:
    pair = pair.strip()
    if not pair:
        continue
    p = pair.splitlines()
    if len(p) != 2:
        continue
    src, trg = tuple(p)
    src_l = ([t.strip() for t in src.split()])
    trg_l = ([t.strip() for t in trg.split()])
    src_vocab = src_vocab.union(set(src_l))
    trg_vocab = trg_vocab.union(set(trg_l))

print('reading English pretrained embedding file...')
file_path2 = 'eng.Skip100'
src_word2vec = dict()
with open(file_path2) as file_word2vec:
    next(file_word2vec)

    file_word2vec_str = file_word2vec.read()
    word_num_pair = file_word2vec_str.split("\n")

    # print(word_num_pair)
    for i in range(0, len(word_num_pair)):
        line = word_num_pair[i].strip()
        if not line:
            continue
        line = line.split()
        word = line[0]
        num = line[1:]

        src_word2vec[word] = torch.FloatTensor([float(item) for item in num])

# print('reading French pretrained embedding file...')
# file_path2 = 'cc.fr.300.vec.gz'
# trg_word2vec = dict()
# with gzip.open(file_path2, 'rb') as file_word2vec:
#     next(file_word2vec)
#
#     for line in file_word2vec:
#         line = str(line, 'utf-8').strip()
#         line = line.split()
#         word = line[0]
#         num = line[1:]
#
#         trg_word2vec[word] = torch.FloatTensor([float(item) for item in num])

trg_word2vec = src_word2vec

#
# construct mapping dict
#
print('constructing mapping dict...')
src_word2id_dict = dict()
src_id2word_dict = dict()
src_vocab.add('<SOS>')
src_vocab.add('<EOS>')
index = 0
for v in src_vocab:
    src_word2id_dict[v] = index
    src_id2word_dict[index] = v
    index += 1
src_SOS_token = src_word2id_dict['<SOS>']
src_EOS_token = src_word2id_dict['<EOS>']

trg_word2id_dict = dict()
trg_id2word_dict = dict()
trg_vocab.add('<SOS>')
trg_vocab.add('<EOS>')
index = 0
for v in trg_vocab:
    trg_word2id_dict[v] = index
    trg_id2word_dict[index] = v
    index += 1
trg_SOS_token = trg_word2id_dict['<SOS>']
trg_EOS_token = trg_word2id_dict['<EOS>']

#
# title abstract pair processing
#
print('title abstarct pair processing...')
file2 = open(file_path)
file2_str = file2.read()

paper = file2_str.split("\n\n")

title_abstract_pair = []
for i in range(0, len(paper)):
    p = paper[i].split('\n')
    if len(p) != 2:
        continue
    title_abstract_pair.append(p)
print(len(title_abstract_pair), "title_abstract_pair loaded.")

index_in_title_abstract_pair = []
for i in range(0, len(title_abstract_pair)):
    pair = []
    title = title_abstract_pair[i][0]
    abstract = title_abstract_pair[i][1]
    title_words = [t.strip() for t in title.split()]
    abstract_words = [t.strip() for t in abstract.split()]

    title_index = []
    for j in range(0, len(title_words)):
        t_w = title_words[j]
        t_w_index = src_word2id_dict[t_w]
        title_index.append(t_w_index)
    title_index.append(src_word2id_dict['<EOS>'])
    title_index = torch.LongTensor(title_index)

    # print(title_index)

    abstract_index = []
    for q in range(0, len(abstract_words)):
        a_w = abstract_words[q]
        a_w_index = trg_word2id_dict[a_w]
        abstract_index.append(a_w_index)
    abstract_index.append(trg_word2id_dict['<EOS>'])
    abstract_index = torch.LongTensor(abstract_index)

    # print(abstract_index)

    # z = zip(title_index, abstract_index)
    # print(list(z))
    index_in_title_abstract_pair.append((title_index, abstract_index))
# index_in_title_abstract_pair = index_in_title_abstract_pair[:15]
# print(index_in_title_abstract_pair)

index_in_title_abstract_pair_train = index_in_title_abstract_pair[:len(index_in_title_abstract_pair)-2000]
index_in_title_abstract_pair_test = index_in_title_abstract_pair[-2000:]

# index_in_title_abstract_pair_train = index_in_title_abstract_pair
# index_in_title_abstract_pair_test = index_in_title_abstract_pair

# for z in range(0, len(title_index)):
#     index_in_title_abstract_pair[z] = (title_index[z], abstract_index[z])
#
# print(index_in_title_abstract_pair)


# encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, gru_hidden_size):
        super(Encoder, self).__init__()
        self.word_embedding_size = word_embedding_size
        self.gru_hidden_size = gru_hidden_size
        self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.gru = nn.GRU(word_embedding_size, gru_hidden_size)

        for p in self.named_parameters():
            if 'bias' in p[0]:
                p[1].data.zero_()
            elif p[1].dim() == 1:
                nn.init.uniform_(p[1].data)
            else:
                nn.init.xavier_uniform_(p[1].data)

    def forward(self, word_embedding, gru_hidden):
        embedded = self.embedding(word_embedding).view(1, 1, -1)
        output = embedded
        output, gru_hidden = self.gru(output, gru_hidden)
        return output, gru_hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.word_embedding_size).cuda()


# # decoder
# class Decoder(nn.Module):
#     def __init__(self, vocab_size, word_embedding_size, gru_hidden_size):
#         super(Decoder, self).__init__()
#         self.word_embedding_size = word_embedding_size
#         self.embedding = nn.Embedding(vocab_size, word_embedding_size)
#         self.gru = nn.GRU(word_embedding_size, gru_hidden_size)
#         self.out = nn.Linear(gru_hidden_size, vocab_size)
#         self.softmax = nn.LogSoftmax(dim=1)
#
#         for p in self.named_parameters():
#             if 'bias' in p[0]:
#                 p[1].data.zero_()
#             elif p[1].dim() == 1:
#                 nn.init.uniform_(p[1].data)
#             else:
#                 nn.init.xavier_uniform_(p[1].data)
#
#     def forward(self, word_embedding, gru_hidden):
#         output = self.embedding(word_embedding).view(1, 1, -1)
#         output = nn.ReLU()(output)
#         output, hidden = self.gru(output, gru_hidden)
#         output = self.softmax(self.out(output[0]))
#         return output, gru_hidden
#
#     def initHidden(self):
#         return torch.zeros(1, 1, self.word_embedding_size).cuda()

#attention decoder
class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


# training
teacher_forcing_ratio = 1


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length= MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.gru_hidden_size).cuda()

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder.forward(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[trg_SOS_token]]).cuda()

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        decoded_words = []
        for di in range(target_length):
            decoder_output, decoder_hidden, attn_weights = decoder.forward(decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.data.topk(1)
            decoded_words.append(trg_id2word_dict[topi.item()])

            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
            decoder_input = target_tensor[di]  # Teacher forcing
        # print(decoded_words)
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, attn_weights = decoder.forward(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
            if decoder_input.item() == trg_EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# timer helper function
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


# time iteration
def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [index_in_title_abstract_pair_train[random.randint(0, len(index_in_title_abstract_pair_train)-1)]
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        input_tensor = input_tensor.cuda()
        target_tensor = target_tensor.cuda()

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if iter % 1000 == 0:
            evaluateRandomly(encoder, decoder)

    showPlot(plot_losses)


# plot data
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# evaluation
def evaluate(encoder, decoder, sentence, max_length = MAX_LENGTH):
    with torch.no_grad():
        input_tensor = sentence
        input_tensor = input_tensor.cuda()
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.gru_hidden_size).cuda()

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[trg_SOS_token]]).cuda()  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []

        for di in range(max_length):
            decoder_output, decoder_hidden, attn_weights = decoder.forward(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == trg_EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(trg_id2word_dict[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = index_in_title_abstract_pair_test[random.randint(0, len(index_in_title_abstract_pair_test)-1)]
        print('>', ' '.join([src_id2word_dict[item.item()] for item in pair[0]]))
        print('=', ' '.join([trg_id2word_dict[item.item()] for item in pair[1]]))
        output_words = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def load_pretrained(encoder, decoder, src_pretrained_word_emb, trg_pretrained_word_emb):
    src_num_pretrained_word = 0
    src_num_word_loaded = 0
    for k, v in src_pretrained_word_emb.items():
        src_num_pretrained_word += 1
        if k not in src_word2id_dict and k.lower() not in src_id2word_dict:
            continue
        if k in src_word2id_dict:
            w_idx = src_word2id_dict[k]
        else:
            w_idx = src_word2id_dict[k.lower()]
        encoder.embedding.weight[w_idx].data = v.data
        src_num_word_loaded += 1

    print('src_num unique words in data', len(src_word2id_dict))
    print('src_num_pretrained_word', src_num_pretrained_word)
    print('src_num_word_loaded', src_num_word_loaded)

    trg_num_pretrained_word = 0
    trg_num_word_loaded = 0
    for k, v in trg_pretrained_word_emb.items():
        trg_num_pretrained_word += 1
        if k not in trg_word2id_dict:
            continue
        w_idx = trg_word2id_dict[k]
        decoder.embedding.weight[w_idx].data = v.data
        trg_num_word_loaded += 1

    print('trg_num unique words in data', len(trg_word2id_dict))
    print('trg_num_pretrained_word', trg_num_pretrained_word)
    print('trg_num_word_loaded', trg_num_word_loaded)

    return encoder, decoder

src_emb_size = 100
trg_emb_size = 300
hidden_size = 100
encoder = Encoder(len(src_word2id_dict), src_emb_size, hidden_size)
decoder = AttnDecoder(hidden_size, len(trg_word2id_dict), dropout_p=0.5)
encoder.cuda()
decoder.cuda()

print('loading pretrained embeddings...')
encoder, decoder = load_pretrained(encoder, decoder, src_word2vec, trg_word2vec)

print('training starts...')
trainIters(encoder, decoder, 500000, print_every=100, learning_rate=0.01)

