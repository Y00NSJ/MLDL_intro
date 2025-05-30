import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree


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

### Logistic Regression
# lr = LogisticRegression()
# lr.fit(train_scaled, train_target)
# print("train score: ", lr.score(train_scaled, train_target))
# print("test score: ", lr.score(test_scaled, test_target))

### Decision Tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print("train score: ", dt.score(train_input, train_target))
print("test score: ", dt.score(test_input, test_target))
print(f"""feature importance: 
['alcohol' 'sugar' 'pH']
{dt.feature_importances_}
""")

# visualization
def visualize(dt):
    plt.figure(figsize=(20, 15))
    plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])   # 결정 트리 시각화
    plt.show()

visualize(dt)