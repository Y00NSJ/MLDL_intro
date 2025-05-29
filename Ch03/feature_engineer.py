import pandas as pd
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
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print("Score of Train Set: ", ridge.score(train_scaled, train_target))
print("Score of Test Set: ", ridge.score(test_scaled, test_target))