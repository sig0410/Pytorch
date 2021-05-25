import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
import jovian
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error

x = torch.tensor([[1, 2, 12, 34, 56, 78, 90, 80],
                  [12, 45, 99, 67, 6, 23, 77, 82],
                  [3, 24, 6, 99, 12, 56, 21, 22]])


model1 = nn.Embedding(100, 7, padding_idx = 0)
## maxlen : 100, 하나의 값에 대한 벡터 차원 : 7
## Embedding Layer 구축
model2 = nn.LSTM(input_size = 7, hidden_size = 3, num_layers = 1, batch_first = True )
# input 3*8*7
# output 3*3

out1 = model1(x)
out2 = model2(out1)

# print(out1.shape)
print(len(out1[1]))
## 각 단어에 대해 고유한 벡터값 부여
# print(out1)
# print(out2)
print(model2(out1))
out, (ht, ct) = model2(out1)
print(ht)
# 3*8 -> embedding -> 3*8*7 -> LSTM -> 3*3

model3 = nn.Sequential(nn.Embedding(100, 7, padding_idx = 0),
                       nn.LSTM(input_size = 7, hidden_size = 3, num_layers = 1, batch_first = True))

out, (hy, ct) = model3(x)
print(out)

# Data Loading
reviews = pd.read_csv('./Womens Clothing E-Commerce Reviews.csv')
print(reviews.shape)
print(reviews.head())

reviews['Title'] = reviews['Title'].fillna('')
reviews['Review Text'] = reviews['Review Text'].fillna('')
# 결측치 채우기

reviews['review'] = reviews['Title'] + ' ' + reviews['Review Text']
# 새로운 컬럼 생성

reviews = reviews[['review', 'Rating']]
# 분석에 활용할 데이터 만들기
reviews.columns = ['review', 'rating']
# 컬럼명 지정
reviews['review_length'] = reviews['review'].apply(lambda x: len(x.split()))
# 텍스트 길이
print(reviews.head())

zero_numbering = {1:0, 2:1, 3:2, 4:3, 5:4}
# 0을 1로 1를 2로 ~~~...

reviews['rating'] = reviews['rating'].apply(lambda x: zero_numbering[x])

print(np.mean(reviews['review_length']))
# 문장 길이 평균


# Tokenization

tok = spacy.load('en')
# 분석기
def tokenize(text):
    text = re.sub(r'[^\x00-\x7F]+', " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    # 구두점이나 숫자 제거
    nopunct = regex.sub(' ', text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]

counts = Counter()
for index, row in reviews.iterrows():
    counts.update(tokenize(row['review']))

# print(counts)
# Base of Frequency for Cleaning Word
print('num_words before: ', len(counts.keys()))
# 추출된 토큰의 개수

for word in list(counts):
    if counts[word] < 2:
        del counts[word]

print('num_words after : ', len(counts.keys()))
# 2이상의 토큰만 추출했을때

vocab2index = {"":0, 'UNK' : 1}
# 빈칸이거나 UNK인 것들은 인덱스를 0과 1로 지정
words = ["", 'UNK']

for word in counts:
    vocab2index[word] = len(words)
    words.append(word)
    # "", UNK를 넣어주는 작업

# print(words)

def encode_sentence(text, vocab2index, N = 70):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype = int)
    # N의 개수만큼 0으로 채워진 벡터
    enc1 = np.array([vocab2index.get(word, vocab2index['UNK']) for word in tokenized])
    length = min(N, len(enc1))
    # 문장 길이가 N보다 작으면 문장 길이 N이 문장 길이보다 작으면 N
    encoded[:length] = enc1[:length]
    # 텍스트가 고유한 인덱스로 구성된 것을 encoded에 length만큼 저장
    return encoded, length


reviews['encoded'] = reviews['review'].apply(lambda x: np.array(encode_sentence(x, vocab2index)))
# reviews에 있는 텍스트를 encoding한 것을 컬럼으로 생성

X = list(reviews['encoded'])
y = list(reviews['rating'])
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2)



class ReviewDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]


train_ds = ReviewDataset(X_train, y_train)
valid_ds = ReviewDataset(X_valid, y_valid)


def train_model(model, epochs = 10, lr = 0.001):
    parameters = filter(lambda p : p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr = lr)

    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0

        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        if i % 5 == 1:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))

def validation_metrics(model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0

    for x, y, l in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total

batch_size = 300
vocab_size = len(words)
train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle = True)
val_dl = DataLoader(valid_ds, batch_size = batch_size)


class LSTM_fixed_len(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, l):
        x = self.embeddings(x)
        x = self.dropout(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])

model_fixed = LSTM_fixed_len(vocab_size, 70, 50 )


train_model(model_fixed, epochs = 30, lr = 0.01)


class LSTM_variable_input(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.3)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 5)

    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first = True, enforce_sorted = False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.linear(ht[-1])
        return out


model = LSTM_variable_input(vocab_size, 70, 50 )

train_model(model, epochs = 30, lr = 0.01)



