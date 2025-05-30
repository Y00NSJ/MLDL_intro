import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# read CSV
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[list(filter(lambda x : x != 'class', wine))]
target = wine['class']

# split train set / test set
train_input, test_input, train_target, test_target = (
    train_test_split(data, target, test_size=0.2, random_state=42)) # 샘플 수가 충분히 많으므로 20%만 테스트셋으로 분리

# preprocess
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# Logistic Regression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print("train score: ", lr.score(train_scaled, train_target))
print("test score: ", lr.score(test_scaled, test_target))