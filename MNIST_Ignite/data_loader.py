import torch

from torch.utils.data import Dataset, DataLoader


class MnistDataset(Dataset): # pytorch의 데이터셋을 상속받음 

    def __init__(self, data, labels, flatten=True):
        self.data = data
        self.labels = labels
        self.flatten = flatten
        # 28*28이므로 flatten해줘야함 , Fully Connected Layer이기 때문에 Flatten 해줘야함 

        super().__init__()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx): 
        x = self.data[idx]
        # x = (28,28)
        y = self.labels[idx]
        # y = (1,)

        if self.flatten:
            x = x.view(-1)
        # x = (784,)
        return x, y
        # if batch size = 256, -> (784,) * 256 => (256,784)


def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y


def get_loaders(config):
    x, y = load_mnist(is_train=True, flatten=False) # is_train : train셋이냐 test셋이냐 

    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt
    # train과 valid셋을 나누는 과정 

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    train_x, valid_x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    # x = (60000, 28, 28) x.size(0) = 60000 이므로 6만개에 대해 랜덤하게 인덱스 shuffle
    # train_x = (48000, 28, 28), valid_x = (12000, 28, 28)
    train_y, valid_y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    # 랜덤 수열로 인덱스를 섞어서 쌍으로 랜덤하게 shuffle

    train_loader = DataLoader(
        dataset=MnistDataset(train_x, train_y, flatten=True),
        batch_size=config.batch_size, # batch_size를 미리 지정 
        shuffle=True, # train은 무조건 shuffling해야함 
    )
    valid_loader = DataLoader(
        dataset=MnistDataset(valid_x, valid_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=True, # valid는 shuffling해도 되고 안해도 됨 
    )

    test_x, test_y = load_mnist(is_train=False, flatten=False)
    test_loader = DataLoader(
        dataset=MnistDataset(test_x, test_y, flatten=True),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader
