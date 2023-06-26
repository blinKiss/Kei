import pandas as pd
import numpy as np

from konlpy.tag import Okt

# 추가 라이브러리
train_data = pd.read_table('./Kiroro/data/ratings_train.txt')
test_data = pd.read_table('./Kiroro/data/ratings_test.txt')
print(train_data[:5])
print(test_data[:5])

print(train_data['document'].nunique())
train_data_uni = train_data['document'].drop_duplicates()
print('훈련용 데이터(중복제거)\n', train_data_uni[0:2])

#긍정과 부정의 비율
print(train_data.groupby('label').size().reset_index(name='count'))
#널값 유무 및 제거
# print(train_data.isnull().values.any())
# print(train_data.isnull().sum())
# print(train_data.loc[train_data.document.isnull()]) # 25857번
train_data = train_data.dropna(how='any')
print(len(train_data))

# 한글과 공백외에 특수문자 제거
train_data['document']=train_data['document'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', regex=True)
# print(train_data[:5])

# 한글과 공백만 남게 되는데 만약 빈데이터가 있으면 널데이터
train_data['document'] = train_data['document'].str.replace('^ +', '', regex=True)
train_data['document'].replace('', np.nan, inplace=True)
# print(train_data.loc[train_data.document.isnull()])
train_data = train_data.dropna(how='any')
print('전처리 후 훈련용 샘플의 개수 :', len(train_data))

# document 열에서 중복인 내용이 있다면 중복 제거
test_data.drop_duplicates(subset = ['document'], inplace=True) 
# 정규 표현식 수행
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", regex=True) 
# 공백은 empty 값으로 변경
test_data['document'] = test_data['document'].str.replace('^ +', "", regex=True) 
test_data['document'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
print('전처리 후 테스트용 샘플의 개수 :',len(test_data)) 

# 2단계 한글문장 토큰화 작업
# 불용어 = 조사 또는 접속사가 대부분
stopword = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

X_train = [] # train_data에서 토큰화 시킨 단어들

from tqdm import tqdm

okt = Okt()
for sentence in tqdm(train_data['document']):
    token_sentence = okt.morphs(sentence, stem=True) # 토큰화 작업 **
    X_train.append(token_sentence)
    
print(X_train[:5])