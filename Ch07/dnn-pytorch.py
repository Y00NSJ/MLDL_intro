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
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
if device == torch.device('mps'): print("Apple Silicon의 MPS 사용")
model = model.to(device)

# 손실 함수 및 옵티마이저 준비
criterion = nn.CrossEntropyLoss()       # 소프트맥스 함수 계산 + 크로스 엔트로피 계산이 합쳐져 있음 -> 출력층에 소프트맥스 추가하지 않아도 됨
optimizer = optim.Adam(model.parameters())  # 최적화시킬 파이토치 텐서 전달; 훈련 가능한 모든 파라미터 전달(제너레이터 객체 호출)하는 메서드 사용

## 훈련
epochs = 5
batches = int(len(train_scaled) / 32)   # 샘플 차원의 크기 / 32 = 배치 횟수
# loop: 에포크 반복
for epoch in range(epochs):
    # 에포크 손실 변수 초기화
    model.train()       # 모델을 훈련 모드로 설정; 특정 층이 훈련할 때/평가할 때 각각 다르게 동작하므로 명시 필요
    train_loss = 0      # 훈련 손실 기록 변수
    # loop: 미니 배치 반복 (경사 하강법 진행)
    for i in range(batches):
        # '배치 입력' 및 타겟 준비
        inputs = train_scaled[i*32:(i+1)*32].to(device)     # 배치 데이터(32개씩) 덜어냄 -> 결과 텐서를 GPU에 적재
        targets = train_target[i*32:(i+1)*32].to(device)
        # 옵티마이저의 그래디언트(손실 함수의 정상에서 내려가야 할 방향과 크기를 알려주는 값) 초기화; 배치마다 새로이 계산해야 하므로
        optimizer.zero_grad()
        # forward pass=forward propagation: 모델에 입력 전달해 출력 생성
        outputs = model(inputs)
        # '모델의 출력' + 타깃을 손실 함수에 전달해 손실 계산
        loss = criterion(outputs, targets)  # 배치에 있는 샘플에 대한 손실의 평균
        # 손실 역전파: 손실을 출력층 -> 입력층으로 거꾸로 전달해 각 층의 모델 파라미터에 대한 그레디언트 계산
        loss.backward()
        # 모델 파라미터 업데이트: 계산된 그레디언트를 사용해 손실함수가 감소되는 방향으로 개선
        optimizer.step()
        # 에포크 손실 기록
        train_loss += loss.item()   # 스칼라를 가진 텐서 객체를 파이썬 타입으로 변환
    # 에포크 손실 출력
    print(f"에포크: {epoch + 1}, 손실: {train_loss/batches:.4f}")

### 검증 셋을 사용해 모델 평가
# 모델을 평가 모드로 설정
model.eval()
# 훈련이 아니므로 그래디언트 계산 제외: 메모리 및 계산량 절약
with torch.no_grad():
    # 검증 셋 및 타겟을 GPU 적재
    val_scaled = val_scaled.to(device)
    val_target = val_target.to(device)
    # 모델 출력 계산
    outputs = model(val_scaled)     # 검증 셋의 샘플 12,000개에 대해 타깃 클래스마다 출력한 값
    # 각 샘플마다 가장 큰 값의 인덱스 추출 => 예측 클래스 생성
    predicts = torch.argmax(outputs, 1)     # 두 번째 축을 따라가 가장 큰 값의 인덱스를 저장
    # 예측 클래스와 val_target 비교 후 -> 올바르게 예측한 개수를 세어 저장
    corrects = (predicts == val_target).sum().item()

# 정확도 계산
accuracy = corrects / len(val_target)
print(f"검증 정확도: {accuracy:.4f}")