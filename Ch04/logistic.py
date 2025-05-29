import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

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