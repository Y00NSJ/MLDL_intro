import pandas as pd


wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[list(filter(lambda x : x != 'class', wine))]
target = wine['class']