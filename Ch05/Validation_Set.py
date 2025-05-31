import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


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
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print("sub-training score: ", dt.score(sub_input, sub_target))
print("validating score: ", dt.score(val_input, val_target))