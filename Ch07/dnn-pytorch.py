from torchvision.datasets import FashionMNIST
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torchinfo import summary
import torch
import torch.optim as optim


### 데이터 다운로드
fm_train = FashionMNIST(root='.', train=True, download=True)    # 저장될 위치, 훈련/테스트 선택, 다운로드해 로컬에 저장
fm_test = FashionMNIST(root='.', train=False, download=True)
type(fm_train.data) # 데이터는 객체의 data 속성에 PyTorch Tensor로 저장됨     # Tensor: PyTorch의 기본 데이터 구조
print(fm_train.targets.shape, fm_test.targets.shape)  # 타깃은 1차원 배열(원-핫 인코딩 X)

### 전처리
train_input = fm_train.data
train_target = fm_train.targets
train_scaled = train_input / 255.0  # 입력 정규화
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

### 모델 생성
model = nn.Sequential(                          # 케라스와 달리 모델 입력 크기 사전 지정 X
    nn.Flatten(),
    nn.Linear(784, 100),  # 케라스의 Dense 층; 매개변수로 입력 크기와 출력 크기(뉴런 개수) 전달
    nn.ReLU(),                                  # 활성 함수는 별도 층으로 추가
    nn.Linear(100, 10)    # 출력층의 활성함수 생략
)

summary(model, input_size=(32, 28, 28))     # 배치 크기를 32로 가정(한 번에 32개의 샘플이 모델에 입력

### 모델 훈련
# 모델을 GPU로 이동 (PyTorch는 명시 지정해 GPU에서 수행할 연산을 구체적으로 제어 가능(유연))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 손실 함수 및 옵티마이저 준비
criterion = nn.CrossEntropyLoss()       # 소프트맥스 함수 계산 + 크로스 엔트로피 계산이 합쳐져 있음 -> 출력층에 소프트맥스 추가하지 않아도 됨
optimizer = optim.Adam(model.parameters())  # 최적화시킬 파이토치 텐서 전달; 훈련 가능한 모든 파라미터 전달(제너레이터 객체 호출)하는 메서드 사용