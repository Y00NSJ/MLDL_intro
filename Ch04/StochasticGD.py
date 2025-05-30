import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 데이터 불러오기
fish = pd.read_csv('https://bit.ly/fish_csv/data')
fish_input = fish[list(filter(lambda x : x != 'Species', fish))]
fish_target = fish['Species']

# train set - test set 분리
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)

# 표준화 전처리