from konlpy.tag import Okt

sentence = '김청길 감독이 만든 이 영화 정말 재밌게 관람했다'
# sentence = '아버지가방에들어가신다'
okt = Okt() # 클래스를 써서 객체변수 선언

# word = okt.morphs(sentence)
token_word = okt.morphs(sentence)
# print(word)
stopword = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']
removed_word = [word for word in token_word if not word in stopword]
print(removed_word)

import pandas as pd
# 판다스 데이터프레임에 중복되는 데이터를 제거
sam_data = {'word' : ['포도', '수박', '포도', '딸기', '수박']}
df = pd.DataFrame(sam_data)
count = df['word'].nunique()
print(count)

df_uni = df['word'].drop_duplicates()
print(df_uni)