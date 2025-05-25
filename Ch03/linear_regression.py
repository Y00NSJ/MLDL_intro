import numpy as np
from perch_data import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt


train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input, train_target)

perch_50 = [[50]]
print("50cm 농어 예측 결과:", knr.predict(perch_50))                  # 실제 무게보다 훨씬 적게 예측
distances, indexes = knr.kneighbors(perch_50)                      # 50cm 농어의 이웃
print("이웃 샘플의 타깃 평균", np.mean(train_target[indexes]))          # 50cm라는 샘플이 훈련셋의 범위를 벗어나 엉뚱한 값이 예측된 것
print("100cm 농어 예측 결과:",knr.predict([[100]]))

plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
plt.scatter(50, 1033, marker='^')                           # 50cm 농어 데이터
plt.scatter(100, 1033, marker='^', color='yellow')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()