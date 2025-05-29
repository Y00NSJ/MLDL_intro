import pandas as pd


fish = pd.read_csv('https://bit.ly/fish_csv_data')
print(fish.columns.tolist()) # 특성 5개
print("어종: ", pd.unique(fish['Species']))   # Species 열의 고유값 측정

fish_input = fish[list(filter(lambda x: x != 'Species', fish))] # 어종을 제외한 5개 열을 입력 데이터로 사용
print(fish_input.head())    # 새롭게 반환된 데이터프레임