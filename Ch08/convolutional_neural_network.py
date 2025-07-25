import keras


## 2차원 입력의 왼쪽 위에서 오른쪽 아래로 이동하는 합성곱 층
keras.layers.Conv2D(10, kernal_size=(3, 3), activation='relu', padding='same', strides=1)  # 필터 개수, 커널 크기
## Max Pooling 수행
keras.layers.MaxPooling2D(2) # 풀링 크기