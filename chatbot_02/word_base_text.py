import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!'
]

myToken = Tokenizer(num_words=100, oov_token='<OOV>') # 토큰화 되지 않은  단어
myToken.fit_on_texts(sentences)
myWord = myToken.word_index
print(myWord)

sentences_other = [
    'I really love my dog',
    'my dog loves my friend'
]

# sequence = myToken.texts_to_sequences(sentences)
sequence_other = myToken.texts_to_sequences(sentences_other)
print(sequence_other)