import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


### 데이터 전처리
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)

### 모델 생성 함수 정의
def model_fn(a_layer=None):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(28, 28)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100, activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

# ## 3개 층 모델 생성
# model = model_fn()
# model.summary()
#
# # 모델 훈련 후 결과 history 객체 확인
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(train_scaled, train_target, epochs=20, verbose=0, validation_data=(val_scaled, val_target))
# # print(history.history.keys())
# # 에포크 별 손실 도식화
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='val')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.show()
# # 에포크 별 정확도 도식화
# plt.plot(history.history['accuracy'])
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.legend()
# plt.show()

### 드롭아웃 층 추가
model = model_fn(keras.layers.Dropout(0.3))
model.summary()

# 평가/예측 시엔 자동으로 드롭아웃 미적용
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_scaled, train_target, epochs=11, verbose=0, validation_data=(val_scaled, val_target))
# 에포크 별 손실 도식화
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# 모델 구조 및 파라미터 저장
model.save('model-whole.keras')
# 훈련된 모델의 파라미터만 저장
model.save_weights('model.weights.h5')

### 모델 생성 후 훈련 생략, 저장했던 파일에서 파라미터 읽어 사용
model_wo_train = model_fn(keras.layers.Dropout(0.3))
model.load_weights('model.weights.h5')      # 정확히 같은 구조의 모델이어야만 불러오기 가능

# 검증 정확도 확인; 12000개 샘플의 10개 클래스에 대한 확률을 모두 확인하는 대신, 10개 중 가장 큰 값의 인덱스만 골라 타겟 레이블과 비교
val_labels = np.argmax(model.predict(val_scaled), axis=-1)  # 새로운 데이터에 대한 정확도만 계산하므로 compile, evaluate 미사용
print(np.mean(val_labels == val_target))    # 인덱스-타깃 값 같으면 1(맞음), 다르면 0(틀림)