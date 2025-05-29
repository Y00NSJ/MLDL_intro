import pandas as pd
from sklearn.model_selection import train_test_split

perch_full = pd.read_csv('https://bit.ly/perch_csv_data')   # 주소로부터 CSV 파일 읽어오기
from perch_data import perch_weight

train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
