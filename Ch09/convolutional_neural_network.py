import keras


## 2차원 입력의 왼쪽 위에서 오른쪽 아래로 이동하는 합성곱 층
keras.layers.Conv2D(10, kernal_size=(3, 3), activation='relu')  # 필터 개수, 커널 크기, 활성 함수