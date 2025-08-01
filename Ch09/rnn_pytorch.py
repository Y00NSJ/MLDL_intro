from keras.datasets import imdb
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import torch

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