from konlpy.tag import Okt

sentence = '김청길 감독이 만든 이 영화 정말 재밌게 관람했다'
# sentence = '아버지가방에들어가신다'
okt = Okt() # 클래스를 써서 객체변수 선언

word = okt.morphs(sentence)

# print(word)

import pandas as pd
# 판다스 데이터프레임에 중복되는 데이터를 제거
sam_data = {'word' : ['포도', '수박', '포도', '딸기', '수박']}
df = pd.DataFrame(sam_data)
count = df['word'].nunique()
print(count)

df_uni = df['word'].drop_duplicates()
print(df_uni)