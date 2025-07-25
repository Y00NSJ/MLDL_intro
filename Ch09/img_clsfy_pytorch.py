from torchvision.datasets import FashionMNIST
from sklearn.model_selection import train_test_split
import torch.nn as nn


### 전처리
fm_train = FashionMNIST(root='.', train=True, download=True)
fm_test = FashionMNIST(root='.', train=False, download=True)

train_input = fm_train.data
train_target = fm_train.targets
train_scaled = train_input.reshape(-1, 1, 28, 28) / 255.0   # 배치 차원, 채널 차원, 이미지 크기

train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

### CNN 구성
