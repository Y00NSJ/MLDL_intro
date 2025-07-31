from keras.datasets import imdb


### 데이터 적재
# 전체 데이터셋에서 가장 자주 등장하는 단어 200개만 사용
# 텍스트의 길이 제각각 => 비정형 2차원 배열 => 각 리뷰마다 별도의 파이썬 리스트 사용 => 1차원 배열
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=200)  # (25000, ) : (25000, )