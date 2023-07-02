# 문제2
# keras.preprocessing.text 의 Tokenizer 클래스를 사용하여 다음 문장을 토큰화하시오 test2.py
 
# sentences = [
# '나는 음악 정말 좋아합니다',
# '당신은 어떤 음악을 좋아하나요?',
# '우리 같이 뮤지컬 보러 갈까요?'
# ]

from keras.preprocessing.text import Tokenizer

sentences = [
'나는 음악 정말 좋아합니다',
'당신은 어떤 음악을 좋아하나요?',
'우리 같이 뮤지컬 보러 갈까요?'
]

myToken = Tokenizer()
myToken.fit_on_texts(sentences)
myWord = myToken.word_index
print(myWord)