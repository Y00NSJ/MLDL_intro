import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

# 데이터 불러오고 split
perch_full = pd.read_csv('https://bit.ly/perch_csv_data')   # 주소로부터 CSV 파일 읽어오기
from perch_data import perch_weight
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)

### 특성 전처리
poly = PolynomialFeatures(degree=5)
poly.fit(train_input)
train_poly = poly.transform(train_input)
# print(poly.get_feature_names_out())
test_poly = poly.transform(test_input)  # info leak 예방 위해 훈련 세트에 적용한 변환기를 기준으로 테스트셋 변환

### Normalization
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

### Multiple Linear Regression
# lr = LinearRegression()
# lr.fit(train_poly, train_target)
# print("Score of Train Set: ", lr.score(train_poly, train_target))
# print("Score of Test Set: ", lr.score(test_poly, test_target))

### Multiple Ridge Regression
# alpha 값 바꿀 때마다 score() 메서드의 결과 저장
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]

for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_scaled, train_target)
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

plt.plot(alpha_list, train_score)
plt.scatter(alpha_list, train_score)
plt.plot(alpha_list, test_score)
plt.scatter(alpha_list, test_score)
plt.xscale('log')       # x축을 로그 스케일로 표시
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

# 최적 alpha 값으로 훈련
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
print("Score of Train Set: ", ridge.score(train_scaled, train_target))
print("Score of Test Set: ", ridge.score(test_scaled, test_target))