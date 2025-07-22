import keras
from sklearn.model_selection import train_test_split


### 전처리
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0  # 픽셀값을 0~255 범위에서 0~1 사이로 정규화
# train_scaled = train_scaled.reshape(-1, 28*28)  # 2차원 배열을 1차원 배열로 펼침
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)


### 입력층 정의
inputs = keras.layers.Input(shape=(784,))

### 밀집층 정의
# 은닉층: sigmoid 사용
dense1 = keras.layers.Dense(100, activation='sigmoid')
# 출력층: softmax 사용
dense2 = keras.layers.Dense(10, activation='softmax')

### DNN 생성
# model = keras.Sequential([inputs, dense1, dense2])  # 입력층은 맨 앞에, 출력층은 맨 뒤에

# model = keras.Sequential([
#     keras.layers.Input(shape=(784,)),
#     keras.layers.Dense(100, activation='sigmoid', name='은닉층'),
#     keras.layers.Dense(10, activation='softmax', name='출력층')
# ], name='패션 MNIST 모델')

model = keras.Sequential()
model.add(keras.layers.Input(shape=(28,28)))    # 1차원으로 펼친 크기가 아닌 원본 이미지 크기로 압력 지정
model.add(keras.layers.Flatten())               # 1차원으로 펼쳐줌
model.add(keras.layers.Dense(100, activation='relu', name='은닉층'))
model.add(keras.layers.Dense(10, activation='softmax', name='출력층'))

model.summary()

### 생성한 DNN 훈련
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_scaled, train_target, epochs=5)