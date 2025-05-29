import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax


fish = pd.read_csv('https://bit.ly/fish_csv_data')
# print(fish.columns.tolist()) # 특성 5개
# print("어종: ", pd.unique(fish['Species']))   # Species 열의 고유값 측정

fish_input = fish[list(filter(lambda x: x != 'Species', fish))] # 어종을 제외한 5개 열을 입력 데이터로 사용
fish_target = fish['Species']   # Dataframe(X) Series 객체(O)

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


### Binary Classification using Logistic Regression
# bream_smelt_idx = (train_target == 'Bream') | (train_target == 'Smelt')     # T/F 배열(series)
# train_b_s = train_scaled[bream_smelt_idx]
# target_b_s = train_target[bream_smelt_idx]
#
# lr = LogisticRegression()
# lr.fit(train_b_s, target_b_s)
# print(f"""Training Completed
# coefficient: {lr.coef_}
# intercept: {lr.intercept_}
# """)
#
# print("Prediction of head samples of train_b_s: ", lr.predict(train_b_s[:5]))
# decisions = lr.decision_function(train_b_s[:5])     # 양성 클래스에 대한 z값 반환
# print("z values: ", decisions)
# print("outputs of sigmoid: ", expit(decisions))
# print("probabilities: ")
# print(lr.classes_)
# print(lr.predict_proba(train_b_s[:5]))


### Multi-Class Classification
lr = LogisticRegression(C=20, max_iter=1000)    # 규제 완화, 반복 횟수 증가
lr.fit(train_scaled, train_target)
print(f"""Training Completed
size of coefficient: {lr.coef_.shape}
size of intercept: {lr.intercept_.shape}
""")

print("Training score:", lr.score(train_scaled, train_target))
print("Test score:", lr.score(test_scaled, test_target))

print("\nPrediction of head samples of test set:", lr.predict(test_scaled[:5]))
decision = lr.decision_function(test_scaled[:5])
print("each z value for each class: ")
print(np.round(decision, decimals=2))
probabilities = softmax(decision, axis=1)   # axis 지정해 각 샘플에 대한 소프트맥스 계산
print("outpusts of softmax: ")
print(np.round(probabilities, decimals=3))

print("\nprobabilities: ")
proba = lr.predict_proba(test_scaled[:5])
print(lr.classes_)
print(np.round(proba, decimals=3))