# 문제3
# keras.preprocessing.text 의 Tokenizer 클래스를 사용하여 다음 문장의 시퀀스를 만드시오 test3.py
 
# sentences = [
# '주말에 여행가고 싶다',
# '여행할 만한 곳을 추천해줘?',
# '거기는 가 본 것 같아'
# ]

from keras.preprocessing.text import Tokenizer

sentences = [
'주말에 여행가고 싶다',
'여행할 만한 곳을 추천해줘?',
'거기는 가 본 것 같아'
]

myToken = Tokenizer()
myToken.fit_on_texts(sentences)

sequences = myToken.texts_to_sequences(sentences)

print(sequences)