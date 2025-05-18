from fishes import *
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

# 샘플 49개
fish_data = np.column_stack((fish_length, fish_weight))
fish_target = np.concatenate((np.ones(35), np.ones(14)))

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

np.random.seed(42)  # 교재와 같은 결과 얻기 위해 시드 지정함
random_idx = np.arange(49)
np.random.shuffle(random_idx)

# 훈련 세트
train_input = input_arr[random_idx[:35]]
train_target = target_arr[random_idx[:35]]
# 테스트 세트
test_input = input_arr[random_idx[35:]]
test_target = target_arr[random_idx[35:]]

# plt.scatter(train_input[:,0], train_input[:,1])
# plt.scatter(test_input[:,0], test_input[:,1])
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)

score = kn.score(test_input, test_target)
print('KNN score:', score)

prediction = kn.predict(np.array([[30, 600]]))
print('KNN prediction:', prediction)
print('actual target:', 1)