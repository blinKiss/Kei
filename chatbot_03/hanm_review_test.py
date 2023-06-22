import pandas as pd

from konlpy.tag import _okt

# 추가 라이브러리
train_data = pd.read_table('./Kiroro/data/ratings_train.txt')
test_data = pd.read_table('./Kiroro/data/ratings_test.txt')
print(train_data[:5])
print(test_data[:5])

print(train_data['document'].nunique())
train_data_uni = train_data['document'].drop_duplicates()
print('훈련용 데이터(중복제거)\n', train_data_uni[0:2])
