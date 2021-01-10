import argparse

import torch 
import torch.nn as nn
import torch.optim as optim 


from model import ImageClassifier
from trainer import Trainer
from data_loader import get_loaders

def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_fn', required = True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type = float, default = .8)
    p.add_argument('--batch_size', type = int, default = 256)
    p.add_argument('--n_epochs', type = int, default = 20)
    p.add_argument('--verbose', type = int, default = 2)

    config = p.parse_args()

    return config
    # 실제 터미널에서 실행을 할때 설정해주는 파라미터라고 생각 
    # config를 받아줌 구성을 만든다고 생각

def main(config):
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda : %d' % config.gpu_id)

    train_loader, valid_loader, test_loader = get_loaders(config)
    # get_loader를 통해 데이터를 받아옴 

    print('Train', len(train_loader.dataset))
    print('Valid', len(valid_loader.dataset))
    print('Test', len(test_loader.dataset))

    model = ImageClassifier(28**2, 10).to(device) # input = (784)이고 10개의 클래스로 분류할 것 
    optimizer = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss() # Loss Function

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader)
    # ignite를 통해 사용 
    
    # 여기까지 필요한것들을 다 불러옴 

if __name__ == '__main__':
    config = define_argparser()
    main(config)

    # python train.py --model_fn model.pth --batch_size_512