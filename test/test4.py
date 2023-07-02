# 문제4
# 문제 3 만든 시쿼스들을 동일한 길이로 시퀀스를 만드는 코드를 작성하시오 test4.py

# 문제3 가져오기~
from keras.preprocessing.text import Tokenizer

sentences = [
'주말에 여행가고 싶다',
'여행할 만한 곳을 추천해줘?',
'거기는 가 본 것 같아'
]

myToken = Tokenizer()
myToken.fit_on_texts(sentences)

sequences = myToken.texts_to_sequences(sentences)
# ~ 문제3 가져오기

max_length = 0
trunc_type = 'post' 
for sequence in sequences:
    if len(sequence) > max_length:
        max_length = len(sequence)


from keras.utils import pad_sequences
seq_paddeds = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
print(seq_paddeds)