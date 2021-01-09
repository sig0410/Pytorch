import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns = cancer.feature_names)
df['class'] = cancer.target

# print(df.tail())
# Custom Dataset Setting
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

data = torch.from_numpy(df.values).float()
print(data.shape)

x = data[:,:10]
y = data[:,-1:]
print(x.shape, y.shape)
# x는 data에서 10개의 컬럼만 추출 y는 data에서 라벨값만 추출 

ratios = [.6, .2, .2]
# train, valid, test ratio
# print(data.size(0))
# data를 미리 지정해준 비율로 나눠주는 과정 
train_cnt = int(data.size(0) * ratios[0])
valid_cnt = int(data.size(0) * ratios[1])
test_cnt = data.size(0) - train_cnt - valid_cnt
cnts = [train_cnt, valid_cnt, test_cnt]
print("train %d / valid %d / test %d" % (train_cnt, valid_cnt, test_cnt))
# 잘나눠진 것을 확인
indices = torch.randperm(data.size(0))
# randperm : random하게 split해줌 즉, 0~568까지의 랜덤 수열 
x = torch.index_select(x, dim = 0, index = indices)
y = torch.index_select(y, dim = 0, index = indices)
# index_select를 통해 shuffling , x = (569,10) dim = 0 은 569자리를 뜻한다.
x = x.split(cnts, dim = 0)
y = y.split(cnts, dim = 0)
# x,y가 텐서형태의 리스트로 나옴 
for x_i, y_i in zip(x,y):
    print(x_i.size(), y_i.size())

# Set Hyper Parameters 
n_epochs = 1000
batch_size = 128
print_interval = 500
early_stop = 100

# Get DataLoader
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        super().__init__()


    def __len__(self):
        return len(self.data)
        # 데이터 길이 출력 함수
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_loader = DataLoader(
    dataset = CustomDataset(x[0], y[0]),
    batch_size = batch_size,
    shuffle = True,
)

valid_loader = DataLoader(
    dataset = CustomDataset(x[1], y[1]),
    batch_size = batch_size,
    shuffle = False,
)

test_loader = DataLoader(
    dataset = CustomDataset(x[2], y[2]),
    batch_size = batch_size,
    shuffle = False,
)

print('train %d / valid %d / test %d' % (
    len(train_loader.dataset),
    len(valid_loader.dataset),
    len(test_loader.dataset),
))
# 데이터가 잘나눠진것을 확인할 수 있다.

# Build Model & Optimizer 
model = nn.Sequential(
    nn.Linear(x[0].size(-1), 6),
    nn.LeakyReLU(),
    nn.Linear(6, 5),
    nn.LeakyReLU(),
    nn.Linear(5, 4),
    nn.LeakyReLU(),
    nn.Linear(4, 3),
    nn.LeakyReLU(),
    nn.Linear(3, y[0].size(-1)),
    nn.Sigmoid(),
)
print(model)
optimizer = optim.Adam(model.parameters())

# Train 
from copy import deepcopy
lowest_loss = np.inf
best_model = None

lowest_epoch = np.inf


train_history, valid_history = [], []

for i in range(n_epochs):
    model.train()
    
    train_loss, valid_loss = 0, 0
    y_hat = []
    
    for x_i, y_i in train_loader:
        y_hat_i = model(x_i)
        loss = F.binary_cross_entropy(y_hat_i, y_i)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()        
        train_loss += float(loss)
        # float을 씌우지 않으면 메모리를 많이 잡아먹음 
    train_loss = train_loss / len(train_loader)
    # 평균 loss

    model.eval()
    with torch.no_grad():
        valid_loss = 0

        for x_i, y_i in valid_loader:
            y_hat_i = model(x_i)
            loss = F.binary_cross_entropy(y_hat_i, y_i)
            valid_loss += float(loss)

            y_hat += [y_hat_i]
    valid_loss = valid_loss / len(valid_loader)
    train_history += [train_loss]
    valid_history += [valid_loss]

    if (i+1) % print_interval == 0:
        print('Epoch %d : trian loss = %.4e valid_loss = %.4e lowest_loss = %.4e' % (
            i +1,
            train_loss,
            valid_loss,
            lowest_loss,
        ))
    if valid_loss <= lowest_loss:
        lowest_loss = valid_loss
        lowest_epoch = i
    else:
        if early_stop > 0 and lowest_epoch + early_stop < i + 1:
            print('There is no improvement during last %d epochs.' % early_stop)
            break 
print('The Best validation loss from epoch %d : %.4e' % (lowest_epoch + 1, lowest_loss))
# model.load_state_dict(best_model)

plot_from = 2 
plt.figure(figsize = (20, 10))
plt.grid(True)
plt.plot(
    range(plot_from, len(train_history)), train_history[plot_from:],
    range(plot_from, len(valid_history)), valid_history[plot_from:],
)
plt.yscale('log')
plt.show()


test_loss = 0
y_hat = []

model.eval()

with torch.no_grad():
    for x_i, y_i in test_loader:
        y_hat_i = model(x_i)
        loss = F.binary_cross_entropy(y_hat_i, y_i)

        test_loss += loss
        y_hat += [y_hat_i]

test_loss = test_loss / len(test_loader)
y_hat = torch.cat(y_hat, dim = 0)

print('test loss %.4e' % valid_loss)
correct_cnt = (y[2] == (y_hat > .5)).sum()
total_cnt = float(y[2].size(0))

print('Test accuracy %.4f' % (correct_cnt / total_cnt))