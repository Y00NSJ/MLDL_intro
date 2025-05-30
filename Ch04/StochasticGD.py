import pandas as pd
import numpy as np
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

### 확률적 경사 하강법을 통해 다중 분류 진행
sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)   # 손실 함수 종류=로지스틱 손실 함수, 에포크 횟수=10
sc.fit(train_scaled, train_target)
print("Training score:", sc.score(train_scaled, train_target))
print("Test score:", sc.score(test_scaled, test_target))

# 앞선 모델에서, 이어서 훈련 진행
sc.partial_fit(train_scaled, train_target)  # 1에포크 추가 훈련
print("+ 1 epoch")
print("Training score:", sc.score(train_scaled, train_target))
print("Test score:", sc.score(test_scaled, test_target))

# Early Stopping
sc = SGDClassifier(loss='log_loss', random_state=42)
train_score = []
test_score = []
classes = np.unique(train_target)
for _ in range(0, 300):     # 300번 진행
    sc.partial_fit(train_scaled, train_target, classes=classes) # fit() 사용하지 않고 partial_fit()만 사용해 훈련, 클래스 레이블 전달
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))