import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier


# 데이터 불러오기
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish_input = fish[list(filter(lambda x : x != 'Species', fish))]
fish_target = fish['Species']

# train set - test set 분리
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

# 각 데이터셋의 특성에 대해 표준화 전처리
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

# 확률적 경사 하강법을 통해 다중 분류 진행
sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)   # 손실 함수 종류=로지스틱 손실 함수, 에포크 횟수=10
sc.fit(train_scaled, train_target)

print("Training score:", sc.score(train_scaled, train_target))
print("Test score:", sc.score(test_scaled, test_target))