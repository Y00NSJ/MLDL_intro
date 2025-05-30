import pandas as pd


fish = pd.read_csv('https://bit.ly/fish_csv/data')
fish_input = fish[list(filter(lambda x : x != 'Species', fish))]
fish_target = fish['Species']