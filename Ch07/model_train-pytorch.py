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

### 모델 훈련
train_hist = []                  # 에포크별 훈련 손실 기록
val_hist = []                    # 에포크별 검증 손실 기록
patience = 2                     # 검증 손실이 향상될 때까지의 에포크 횟수
best_loss = -1                   # 최상 손실 기록
early_stopping_counter = 0       # 연속적으로 검증 손실이 향상되지 않은 에포크 횟수 기록; patience 이상일 시 종료

epochs = 20
batches = int(len(train_scaled)/32)
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for i in range(batches):
        inputs = train_scaled[i*32:(i+1)*32].to(device)
        targets = train_target[i*32:(i+1)*32].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    # 검증 손실 계산
    model.eval()
    val_loss = 0
    with torch.no_grad():
        val_scaled = val_scaled.to(device)
        val_target = val_target.to(device)
        outputs = model(val_scaled)
        loss = criterion(outputs, val_target)
        val_loss += loss.item()
    train_hist.append(train_loss/batches)
    val_hist.append(val_loss)
    print(f"에포크: {epoch+1}",
          f"훈련 손실: {train_loss/batches:.4f}, 검증 손실: {val_loss:.4f}")
    ## 조기 종료
    if best_loss == -1 or val_loss < best_loss:     # 첫 번째 에포크이거나 검증 손실 이전에 기록된 최상의 손실보다 작으면
        best_loss = val_loss                        # 최상의 손실을 현재 검증 손실로 업데이트
        early_stopping_counter = 0                  # 조기 종료 카운터를 0으로 초기화 => 검증 손실이 더 좋아지지 않더라도 patience 횟수만큼 인내
        torch.save(model.state_dict(), 'best_model.pt')     # 최상의 검증 손실을 산출한 모델을 저장(구조+파라미터)
    else:                                           # 검증 손실이 더 나아지지 않았다면
        early_stopping_counter += 1                 # 카운터 증가
        if early_stopping_counter >= patience:      # 카운터가 patience보다 크면 조기종료
            print(f"{epoch+1}번째 에포크에서 조기 종료되었습니다.")
            break