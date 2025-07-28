import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST

from Ch08.cnn_visualization import ankle_boot

### 동일한 모델 생성 후 가중치 로드
model = nn.Sequential()
model.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, padding='same'))
model.add_module('relu1', nn.ReLU())
model.add_module('pool1', nn.MaxPool2d(2))
model.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding='same'))
model.add_module('relu2', nn.ReLU())
model.add_module('pool2', nn.MaxPool2d(2))
model.add_module('flatten', nn.Flatten())
model.add_module('dense1', nn.Linear(3136, 100))
model.add_module('relu3', nn.ReLU())
model.add_module('dropout', nn.Dropout(0.3))
model.add_module('dense2', nn.Linear(100, 10))
# 가중치 로드
model.load_state_dict(torch.load('best_cnn_model.pt', weights_only=True))

### 층 참조
# - generator 객체 활용
layers = [layer for layer in model.children()]
layers[0]
# - Sequential 클래스 모델: 정수 인덱스로 참조
model[0]
# - 모델 객체의 메서드 활용
for name, layer in model.named_children():
    print(f"{name: 10s}, layer")
# - 층의 이름을 모델의 속성처럼 사용
model.conv1
