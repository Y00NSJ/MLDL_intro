import keras
from sklearn.model_selection import train_test_split


### 전처리
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0  # 픽셀값을 0~255 범위에서 0~1 사이로 정규화
train_scaled = train_scaled.reshape(-1, 28*28)  # 2차원 배열을 1차원 배열로 펼침
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)


### 입력층 정의
inputs = keras.layers.Input(shape=(784,))

### 밀집층 정의
# 은닉층: sigmoid 사용
dense1 = keras.layers.Dense(100, activation='sigmoid')
# 출력층: softmax 사용
dense2 = keras.layers.Dense(10, activation='softmax')