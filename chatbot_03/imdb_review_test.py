import tensorflow_datasets as tfds
import numpy as np

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

# 훈련데이터
train_sentences = []
train_labels = []
for st, lb in train_data:
    train_sentences.append(str(st.numpy()))
    train_labels.append((str(lb.numpy())))

print(train_sentences[0])
print(train_labels[0]) # 0은 부정 1은 긍정
print(len(train_sentences[0]))

# 테스트 데이터
test_sentences = []
test_labels = []
for st, lb in test_data:
    test_sentences.append(str(st.numpy()))
    test_labels.append(str(lb.numpy()))
   
print('\n') 
print(test_sentences[0])
print(test_labels[0])
print(len(test_sentences))

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
vocab_size = 10000
oov_tok = '<OOV>'
max_length = 120
trunc_type = 'post' # 0을 어디에 붙일까?

imdb_token = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
imdb_token.fit_on_texts(train_sentences)
imdb_word_index = imdb_token.word_index

imdb_sequences = imdb_token.texts_to_sequences(train_sentences)
imdb_paddeds = pad_sequences(imdb_sequences, maxlen=max_length, truncating=trunc_type)
print(imdb_sequences[0])
print(imdb_paddeds[0])
