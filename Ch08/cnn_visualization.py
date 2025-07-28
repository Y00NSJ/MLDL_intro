import keras
import matplotlib.pyplot as plt


### img_classification.py에서 훈련했던 CNN의 체크포인트 파일 불러오기
model = keras.models.load_model('best-cnn-model.keras')
# print(model.layers)     # 층 확인

### 첫 번째 합성곱 층의 가중치 조사
conv = model.layers[0]
# print(f"가중치: {conv.weights[0].shape}, 절편: {conv.weights[1].shape}")     # 커널 크기 3*3, 깊이 1, 필터 수 32
conv_weights = conv.weights[0].numpy()
print(f"가중치 평균: {conv_weights.mean()}, 가중치 표준편차: {conv_weights.std()}") # 0에 근접 / 0.23
# 시각화
plt.hist(conv_weights.reshape(-1, 1))   # 1차원 배열로 전달
plt.xlabel('weights')
plt.ylabel('count')
plt.show()

fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[:, :, 0, i*16+j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

### empty CNN 생성
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Input(shape=(28, 28, 1)))
no_training_model.add(keras.layers.Conv2D(32, kernal_size=3, activation='relu', padding='same'))
# 1번 층의 가중치를 변수에 저장
no_training_conv = no_training_model.layers[0]
no_training_weights = no_training_conv.weights[0].numpy()
print(f"가중치 평균: {no_training_weights.mean()}, 가중치 표준편차: {no_training_weights.std()}") # 0에 근접 / 이전 모델보다 매우 작음
# 시각화
plt.hist(no_training_weights.reshape(-1, 1))   # 1차원 배열로 전달
plt.xlabel('weights')
plt.ylabel('count')
plt.show()

fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(no_training_weights[:, :, 0, i*16+j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()


### 함수형 API로 모델 구성
inputs = keras.Input(shape=(784,))
dense1 = keras.layers.Dense(100, activation='relu')
dense2 = keras.layers.Dense(10, activation='softmax')
hidden = dense1(inputs)     # inputs를 Dense1 층에 통과시킨 후 출력값을 hidden에 할당
outputs = dense2(hidden)
func_model = keras.Model(inputs=inputs, outputs=outputs)


### 특성 맵 시각화
# 패션 MNIST 데이터셋 읽어오기
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
plt.imshow(train_input[0], cmap='gray_r')
plt.show()