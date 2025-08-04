from keras.datasets import imdb
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


## 데이터 로드 및 검증 셋 분할
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

## 각 시퀀스의 길이를 100으로 통일
train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)

## 넘파이 배열 -> 파이토치 텐서 변환
# input
train_seq = torch.tensor(train_seq)
val_seq = torch.tensor(val_seq)
# target: PyTorch의 손실 함수는 입력으로 실숫값을 기대하므로 data type 변환도 필요
print("타깃 배열의 데이터 타입: ", train_target.dtype)    # int64: 부정(0) or 긍정(1)
train_target = torch.tensor(train_target, dtype=torch.float32)
val_target = torch.tensor(val_target, dtype=torch.float32)

## 데이터 로더로 훈련셋 및 검증셋 준비
train_dataset = TensorDataset(train_seq, train_target)
val_dataset = TensorDataset(val_seq, val_target)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

## 모델 구현
# Sequential 대신 nn.Module 상속 클래스 생성
class IMDBRnn(nn.Module):
    def __init__(self):
        super().__init__()
        # (어휘 사전 크기(배치 크기), 임베딩 벡터 크기(시퀀스 길이)) -> (배치 크기, 시퀀스 길이, 임베딩 크기)
        self.embedding = nn.Embedding(500, 16)
        # (시퀀스 길이, 배치 크기, 임베딩 크기)_임베딩 층의 출력 순서는 다르므로 매개변수 지정
        self.rnn = nn.RNN(16, 8, batch_first=True)
        # 밀집층
        self.dense = nn.Linear(8, 1)
        # 이진 분류 -> 시그모이드 활성 함수
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 정의된 층을 차례로 호출
        x = self.embedding(x)
        _, hidden = self.rnn(x)             # 반환값 중 은닉 상태는 미사용
        outputs = self.dense(hidden[-1])    # 여러 개의 층을 사용하는 경우를 가정해 마지막 층의 은닉 객체를 선택
        return self.sigmoid(outputs)

## 모델 객체 생성 및 GPU 전달
model = IMDBRnn()

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

## 손실 함수 및 옵티마이저 정의
criterion = nn.BCELoss()    # 마지막 층이 시그모이드였으므로 해당 함수 사용
optimizer = optim.Adam(model.parameters(), lr=2e-4)

## 모델 훈련
train_hist = []
val_hist = []
patience = 2
best_loss = -1
early_stopping_counter = 0

epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()

    train_loss = train_loss/len(train_loader)
    val_loss = val_loss/len(val_loader)
    train_hist.append(train_loss)
    val_hist.append(val_loss)
    print(f"에포크:{epoch+1},",
          f"훈련 손실:{train_loss:.4f}, 검증 손실:{val_loss:.4f}")

    if best_loss == -1 or val_loss < best_loss:
        best_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'best_rnn_model.pt')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print(f"{epoch+1}번째 에포크에서 조기 종료되었습니다.")
            break

## 손실 그래프 시각화
plt.plot(train_hist, label='train')
plt.plot(val_hist, label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

## 검증 셋 활용해 정확도 확인
model.load_state_dict(torch.load('best_rnn_model.pt', weights_only=True))

model.eval()
corrects = 0
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        predicts = outputs > 0.5
        corrects += (predicts.squeeze() == targets).sum().item()

accuracy = corrects / len(val_dataset)
print(f"검증 정확도: {accuracy:.4f}")