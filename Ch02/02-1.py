from fishes import *
from sklearn.neighbors import KNeighborsClassifier


# 샘플 49개
fish_data = [[l, w] for l, w in zip(fish_length, fish_weight)]
fish_target = [1]*35 + [0]*14
# 훈련 세트
train_input = fish_data[:35]
train_target = fish_target[:35]
# 테스트 세트
test_input = fish_data[35:]
test_target = fish_target[35:]

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)

score = kn.score(test_input, test_target)
print('KNN score:', score)