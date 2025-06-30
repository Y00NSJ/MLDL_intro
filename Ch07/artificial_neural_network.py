import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.linear_model import SGDClassifier
import tensorflow as tf

# fetch Fashion MNIST dataset
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
# fig, axs = plt.subplots(1, 10, figsize=(10, 10))    # 1개 행, 10개 열
# for i in range(10):
#     axs[i].imshow(train_input[i], cmap='gray_r')
#     axs[i].axis('off')
# plt.show()
# print(train_target[:10])
# print(np.unique(train_target, return_counts=True))


### Classify using Logistic Regression
# SGD Classifier를 사용하기 위해 각 이미지 샘플을 1차원으로 reshape
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)  # 784

# 로지스틱 회귀 및 교차 검증
# sc = SGDClassifier(loss='log_loss', max_iter=50, random_state=42)
# scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
# print(f"average test score: {np.mean(scores['test_score'])}")


### ANN
# 20% 검증 셋으로 분리
train_scaled, val_scaled, train_target, val_target = (
    train_test_split(train_scaled, train_target, test_size=0.2, random_state=42))

# 입력층 정의
inputs = keras.layers.Input(shape=(784,))
# 밀집층 정의
dense = keras.layers.Dense(10, activation='softmax')    # 뉴런의 출력값을 확률로 변경
# 신경망 모델 생성
model = keras.models.Sequential([inputs, dense])

# 훈련 전 설정
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# 훈련
model.fit(train_scaled, train_target, epochs=5)
# 검증 셋을 통해 성능 평가
model.evaluate(val_scaled, val_target)