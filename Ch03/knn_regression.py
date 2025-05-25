import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from perch_data import *


train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input = train_input.reshape(-1, 1) # 첫 번째 크기를 원소 개수에 맞게 설정
test_input = test_input.reshape(-1, 1)

knn_reg = KNeighborsRegressor()
knn_reg.fit(train_input, train_target)

test_set_score = knn_reg.score(test_input, test_target)
print(f"score of KNN Regressor: {test_set_score}, 테스트셋 사용")

test_prediction = knn_reg.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print (f"타깃과 예측의 절댓값 오차 평균: {mae}")

train_set_score = knn_reg.score(train_input, train_target)
print(f"score of KNN Regressor: {train_set_score}, 훈련셋 사용")


print("\n====== underfitting 이슈 보완 =======\n")
knn_reg.n_neighbors = 3
knn_reg.fit(train_input, train_target)
train_set_score = knn_reg.score(train_input, train_target)
print(f"score of KNN Regressor: {train_set_score}, 훈련셋 사용")
test_set_score = knn_reg.score(test_input, test_target)
print(f"score of KNN Regressor: {test_set_score}, 테스트셋 사용")