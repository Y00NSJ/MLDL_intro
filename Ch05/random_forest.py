import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance


### read CSV
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[list(filter(lambda x : x != 'class', wine))]
target = wine['class']

### split between train set and test set
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)

### train Random Forest and cross validate
# rf = RandomForestClassifier(n_jobs=-1, random_state=42)
# scores = cross_validate(rf, train_input, train_target, return_train_score=True, n_jobs=-1)
# print(f"train score: {np.mean(scores['train_score'])}\ntest score: {np.mean(scores['test_score'])}")  # overfitting
# rf.fit(train_input, train_target)
# print("feature importance:", rf.feature_importances_)   # 하나의 특성에 과집중 (X), 좀 더 다양한 특성이 훈련에 기여 => 일반화 성능 향상

### use OOB sample
# rf = RandomForestClassifier(oob_score=True, n_jobs=-1, random_state=42)
# rf.fit(train_input, train_target)
# print("Evaluation Score using OOB:", rf.oob_score_)

### Extra Trees
# et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
# scores = cross_validate(et, train_input, train_target, return_train_score=True, n_jobs=-1)
# print(f"train score: {np.mean(scores['train_score'])}\ntest score: {np.mean(scores['test_score'])}")
# et.fit(train_input, train_target)
# print("feature importance:", et.feature_importances_)

### Gradient Boosting
# gb = GradientBoostingClassifier(random_state=42)
# scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
# print(f"train score: {np.mean(scores['train_score'])}\ntest score: {np.mean(scores['test_score'])}")
# gb.fit(train_input, train_target)
# print("feature importance:", gb.feature_importances_)

### Hisogram-based Gradient Boosting
hgb = HistGradientBoostingClassifier(random_state=42)
# scores = cross_validate(hgb, train_input, train_target, return_train_score=True)
# print(f"train score: {np.mean(scores['train_score'])}\ntest score: {np.mean(scores['test_score'])}")
hgb.fit(train_input, train_target)
print("Training Terminated")
result = permutation_importance(hgb, train_input, train_target, n_repeats=10, random_state=42, n_jobs=-1)  # 10번 섞음
print("feature importance of Train Set:",result.importances_mean)
test_result = permutation_importance(hgb, test_input, test_target, n_repeats=10, random_state=42, n_jobs=-1)
print("feature importance of Test Set:",test_result.importances_mean)
print("Test Score:", hgb.score(test_input, test_target))