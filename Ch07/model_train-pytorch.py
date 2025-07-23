from sklearn.model_selection import train_test_split
from torchvision.datasets import FashionMNIST
import torch.nn as nn
import torch
import torch.optim as optim


### 데이터 로드 및 전처리
fm_train = FashionMNIST(root='.', train=True, download=True)
fm_test = FashionMNIST(root='.', train=False, download=True)

train_input = fm_train.data
train_target = fm_train.targets
train_scaled = train_input / 255.0  # 입력 정규화
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

### 모델 생성
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 100),
    nn.ReLU(),
    nn.Dropout(0.3),    # 30% 드롭아웃
    nn.Linear(100, 10)
)
# 모델을 GPU에 적재
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
# 손실 함수 및 옵티마이저 준비
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

