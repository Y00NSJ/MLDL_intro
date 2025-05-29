import pandas as pd

perch_full = pd.read_csv('https://bit.ly/perch_csv_data')   # 주소로부터 CSV 파일 읽어오기
print(perch_full.head())                                           # 최초 5개 행 출력
