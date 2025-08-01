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
model.add(keras.layers.Input(shape=(100, 200)))                   # 원-핫 인코딩 결과를 입력
model.add(keras.layers.SimpleRNN(8))                              # 순환층 뉴런 개수 8
model.add(keras.layers.Dense(1, activation='sigmoid'))      # 이진 분류 문제이므로

## 원-핫 인코딩으로 훈련 셋/검증 셋 준비
train_oh = keras.utils.to_categorical(train_seq)
print("train_seq 배열의 원-핫 인코딩 결과: ", train_oh.shape)     # 20000개 시퀀스의 각 100개씩의 단어가 각각 200 크기로 인코딩됨
print("인코딩 결과 예시_0번 샘플의 0번 토큰, 상위 12개 요소: ", train_oh[0][0][:12])
print("인코딩 검증_200개 정수의 합이 1: ", np.sum(train_oh[0][0]))
val_oh = keras.utils.to_categorical(val_seq)

## 모델 구조 확인
model.summary()         # (200차원 입력 * 뉴런 8개) + (은닉 상태 크기 * 뉴런 개수) + 절편 8개


### 순환 신경망 훈련
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(train_oh, train_target,
                    epochs=100, batch_size=64,
                    validation_data=(val_oh, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])

## 훈련 손실, 검증 손실 도식화
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()


### 단어 임베딩 방식
## 데이터셋 준비: 500개 단어까지 사용
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
