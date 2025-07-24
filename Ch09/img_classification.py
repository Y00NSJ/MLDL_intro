import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled =  train_input.reshape(-1, 28, 28, 1) / 255.0      # flatten 없이, 2차원 배열에 차원 추가 -> (28,28,1)*48000개
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

### CNN 만들기
model = keras.Sequential()
# 입력층
model.add(keras.layers.Input(shape=(28, 28, 1)))
# 합성곱 층: (3*3*1 * 32개층) + 32개절편 = 320개 파라미터
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'))
# 풀링 층
model.add(keras.layers.MaxPooling2D(2))     # => (14, 14, 32)
# 2번 합성곱 층
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'))
# 2번 풀링 층
model.add(keras.layers.MaxPooling2D(2))     # => (7, 7, 64)
# Flatten 층
model.add(keras.layers.Flatten())
# 밀집 은닉층(Dense): (7*7*64) * 100개뉴런 + 100개절편
model.add(keras.layers.Dense(100, activation='relu'))
# Dropout 층
model.add(keras.layers.Dropout(0.4))
# 출력층
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
keras.utils.plot_model(model, show_shapes=True)   # 층의 구성 도식화

### 모델 컴파일 및 훈련
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.keras', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)    # cp가 저장한 모델을 불러오지 않아도, 이미 현재 최적 파라미터로 복원되어 있음

history = model.fit(train_scaled, train_target, epochs=20, validation_data=(val_scaled, val_target), callbacks=[checkpoint_cb, early_stopping_cb])

# 손실 그래프 도식화해 조기 종료 여부 확인
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

### 성능 평가
model.evaluate(val_scaled, val_target)

### 새로운 데이터에 대한 예측 진행
# 편의 상 검증 셋의 첫 번째 샘플을 처음 본 이미지라고 가정
preds = model.predict(val_scaled[0:1])  # 슬라이싱은 인덱싱과 다르게 선택된 원소가 1개여도 전체 차원을 유지->입력의 첫 번째 차원은 배치 차원
print("예측 결과: 가방")
print(preds)
# 예측 결과를 막대그래프로 도식화
plt.bar(range(1, 11), preds[0])
plt.xlabel('class')
plt.ylabel('predicted probability')
plt.show()