import numpy as np
from perch_data import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input, train_target)

perch_50 = [[50]]
# print("50cm 농어 예측 결과:", knr.predict(perch_50))                  # 실제 무게보다 훨씬 적게 예측
# distances, indexes = knr.kneighbors(perch_50)                      # 50cm 농어의 이웃
# print("이웃 샘플의 타깃 평균", np.mean(train_target[indexes]))          # 50cm라는 샘플이 훈련셋의 범위를 벗어나 엉뚱한 값이 예측된 것
# print("100cm 농어 예측 결과:",knr.predict([[100]]))

lr = LinearRegression()
lr.fit(train_input, train_target)
print("== LR 모델의 model parameter ==")
print(f"기울기(coefficient): {lr.coef_}\n절편(intercept): {lr.intercept_}\n")

print("LR 모델의 50cm 농어 예측 결과:", lr.predict(perch_50))
plt.scatter(train_input, train_target)                                              # 훈련 셋 산점도
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_], color='green')    # 농어 길이 15 -> 50까지의 1차 방정식 그래프
plt.scatter(50, 1241.8, marker='^', color='red')    # 선형 회귀 모델의 예측 결과


print(f"score of LR: {lr.score(train_input, train_target)}, 훈련셋 사용")
print(f"score of LR: {lr.score(test_input, test_target)}, 테스트셋 사용")


print()
### Polynomial Regression
train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))
pr = LinearRegression()
pr.fit(train_poly, train_target)
print("== PR 모델의 model parameter ==")
print(f"기울기(coefficient): {pr.coef_}\n절편(intercept): {pr.intercept_}\n")

perch_50_poly = np.column_stack((np.array(perch_50)**2, perch_50))
print("PR 모델의 50cm 농어 예측 결과:", pr.predict(perch_50_poly))

print(f"score of PR: {pr.score(train_poly, train_target)}, 훈련셋 사용")
print(f"score of PR: {pr.score(test_poly, test_target)}, 테스트셋 사용")

# 다항 회귀 그래프
point = np.arange(15, 50)
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
plt.scatter(50, 1574, marker='^')   # 다항 회귀 모델의 예측 결과

plt.xlabel('length')
plt.ylabel('weight')
plt.show()