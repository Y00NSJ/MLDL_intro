from keras.datasets import imdb
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import keras


### 데이터 적재
## 전체 데이터셋에서 가장 자주 등장하는 단어 200개만 사용
## 텍스트의 길이 제각각 => 비정형 2차원 배열 => 각 리뷰마다 별도의 파이썬 리스트 사용 => 1차원 배열
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=200)  # (25000, ) : (25000, )
## 첫 번째 리뷰
# print("첫 번째 리뷰의 길이: ", len(train_input[0]))
# print("첫 번째 리뷰의 내용:\n", train_input[0])
# # 타깃 데이터
# print("타깃 데이터는 긍/부정 여부로 구성: ", train_target[:10])

### 데이터 전처리
## 훈련 셋에서 검증 셋 분리
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
## 훈련 셋 조사: 각 리뷰의 길이 통계
# lengths = np.array([len(x) for x in train_input])
# print(f"리뷰 길이의 평균: {np.mean(lengths)}, 중간값: {np.median(lengths)}")
# plt.hist(lengths)
# plt.xlabel('length')
# plt.ylabel('frequency')
# plt.show()
## 리뷰의 길이 고정
train_seq = pad_sequences(train_input, maxlen=100)  # 긴 리뷰는 시퀀스의 "앞 부분"을 자름(truncating=pre); 뒷부분이 더 결정적 정보라고 기대
val_seq = pad_sequences(val_input, maxlen=100)

### 순환 신경망 구성
model = keras.Sequential()
model.add(keras.layers.Input(shape=(100, 200)))     # 원-핫 인코딩
model.add(keras.layers.SimpleRNN(8))
model.add(keras.layers.Dense(1, activation='sigmoid'))      # 이진 분류 문제이므로
