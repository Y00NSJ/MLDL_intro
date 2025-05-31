import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import uniform, randint


# read CSV
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[list(filter(lambda x : x != 'class', wine))]
target = wine['class']

# split train set / test set
train_input, test_input, train_target, test_target = (
    train_test_split(data, target, test_size=0.2, random_state=42))
# split sub-train set / validation set from train set
sub_input, val_input, sub_target, val_target = (
    train_test_split(train_input, train_target, test_size=0.2, random_state=42))


# train and test model w/ sub-train set and validation set
# dt = DecisionTreeClassifier(random_state=42)
# dt.fit(sub_input, sub_target)
# print("sub-training score: ", dt.score(sub_input, sub_target))
# print("validating score: ", dt.score(val_input, val_target))


### cross validation
# 5-fold
# scores = cross_validate(dt, train_input, train_target)
# for score in scores:
#     print(f"{score}: {scores[score]}")
# print("final score: ", np.mean(scores['test_score']))

# 10-fold cv with splitter
# splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# scores = cross_validate(dt, train_input, train_target, cv=splitter)
# print("final score: ", np.mean(scores['test_score']))


### Tune Hyperparameter
# 탐색할 매개변수: 탐색할 값 리스트
# params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),    # 9
#           'max_depth': range(5, 20, 1),                                 # *15
#           'min_samples_split': range(2, 100, 10)}                       # *10
# gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)   # 시스템 내 모든 코어 사용해 병렬 실행
# gs.fit(train_input, train_target)
#
# # train a model w/ the best hyperparameter using whole train set
# dt = gs.best_estimator_
# print("train terminated, final score: ", dt.score(train_input, train_target))
#
# # best hyperparameter and cv scores
# print("best params combination: ", gs.best_params_)                                 # 최적 매개변수
# print("cv scores: ", gs.cv_results_['mean_test_score'])                             # 각 매개변수에서 수행한 교차검증의 평균 점수
# print("best CV score: ", np.max(gs.cv_results_['mean_test_score']))
# # ... can be expressed as...
# print(gs.cv_results_['params'][gs.best_index_])                         # 가장 높은 값의 인덱스 사용해 params 키에 저장된 매개변수 출력


### Random Search
# sampling from uniform distribution
# rgen = randint(0, 10)
# print(np.unique(rgen.rvs(1000), return_counts=True))
# ugen = uniform(0, 1)
# print(ugen.rvs(10))
params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25),
          }

rs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, n_iter=100, n_jobs=-1, random_state=42)
rs.fit(train_input, train_target)

print("best params combination: ", rs.best_params_)
print("best CV score: ", np.max(rs.cv_results_['mean_test_score']))

dt = rs.best_estimator_
print("Train Terminated, Test Score: ", dt.score(test_input, test_target))