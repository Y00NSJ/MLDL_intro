from keras.datasets import imdb
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn


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