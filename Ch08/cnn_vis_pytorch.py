import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST


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

### 가중치의 평균 및 표준편차 계산
conv_weights = model.conv1.weight.data
print(f"가중치 평균: {conv_weights.mean()}, 표준편차: {conv_weights.std()}")
# PyTorch: 가중치 텐서에서 채널 차원이 가장 먼저 나옴; (필터 개수, 채널 차원, 높이, 너비)
fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[i*16 + j,0,:,:], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

### 활성 출력 확인
fm_train = FashionMNIST(root='.', train=True, download=True)
train_input = fm_train.data

## 0번 샘플을 첫 번째 합성곱 층에 전달
# 전처리
ankle_boot = train_input[0:1].reshape(1, 1, 28, 28) / 255.0
# 모델 통과
model.eval()
with torch.no_grad():
    feature_maps = model.conv1(ankle_boot)
    # Keras와 달리 ReLU 함수가 별도 층으로 분리되어 있으므로 적용
    feature_maps = model.relu1(feature_maps)

## 시각화
fig, axs = plt.subplots(4, 8, figsize=(15,8))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,i*8 + j,:,:])
        axs[i, j].axis('off')
plt.show()

## 두 번째 합성곱 층에 전달
model.eval()
# 층이 깊다면 위의 방식이 비효율적이므로 아래 방식 사용
with torch.no_grad():
    for name, layer in model.named_children():
        x = layer(x)
        if name == 'relu2':
            break
feature_maps = x
# 시각화
fig, axs = plt.subplots(8, 8, figsize=(12,12))
for i in range(8):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,i*8 + j,:,:])
        axs[i, j].axis('off')
plt.show()