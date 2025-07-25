from torchvision.datasets import FashionMNIST
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


### 전처리
fm_train = FashionMNIST(root='.', train=True, download=True)
fm_test = FashionMNIST(root='.', train=False, download=True)

train_input = fm_train.data
train_target = fm_train.targets
train_scaled = train_input.reshape(-1, 1, 28, 28) / 255.0   # 배치 차원, 채널 차원, 이미지 크기

train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

### CNN 구성: 모델 객체 생성 후 층을 하나씩 추가
model = nn.Sequential()
# 합성곱 층
model.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, padding='same'))  # 입력채널 수, 출력채널=필터 수
# 활성 함수
model.add_module('relu1', nn.ReLU())
# 풀링 층
model.add_module('pool1', nn.MaxPool2d(2))
# 2nd 합성곱 층
model.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding='same'))
# 2nd 활성 함수
model.add_module('relu2', nn.ReLU())
# 2nd 풀링 층
model.add_module('pool2', nn.MaxPool2d(2))
# Flatten 층 => 7*7*64
model.add_module('flatten', nn.Flatten())
outputs = model(torch.ones(1, 1, 28, 28))   # 값이 모두 1로 채워진 배열을 전달해 출력의 크기 확인
print(outputs.shape)                        # [1, 3136], 1은 배치 차원에 담긴 샘플 개수
# 밀집층
model.add_module('dense1', nn.Linear(7 * 7 * 64, 100))
# 활성 함수
model.add_module('relu3', nn.ReLU())
# 드롭아웃
model.add_module('dropout', nn.Dropout(0.3))
# 출력층
model.add_module('dense2', nn.Linear(100, 10))

### 훈련
# GPU로 모델 전달
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)
# 옵티마이저 준비
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 훈련 셋을 배치 크기로 하드코딩식 나눗셈하는 대신, PyTorch의 Data Loader 활용; 에포크마다 훈련 셋 섞는 기능까지 포함
# 미니 배치 경사 하강법에선 에포크마다 훈련 샘플을 섞음으로써 샘플 순서에서 발생하는 편향 예방
train_dataset = TensorDataset(train_scaled, train_target)
val_dataset = TensorDataset(val_scaled, val_target)