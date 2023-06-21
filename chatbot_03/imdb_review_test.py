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
embedding_dim = 16
max_length = 120
trunc_type = 'post' # 0을 어디에 붙일까?
oov_tok = '<OOV>'

imdb_token = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
imdb_token.fit_on_texts(train_sentences)
imdb_word_index = imdb_token.word_index

imdb_sequences = imdb_token.texts_to_sequences(train_sentences)
# 자연어 처리로 나온 최종 결과
imdb_paddeds = pad_sequences(imdb_sequences, maxlen=max_length, truncating=trunc_type)
# print(imdb_sequences[0])
# print(imdb_paddeds[0])

# 테스트 데이터 시퀀스
test_seqences = imdb_token.texts_to_sequences(test_sentences)
test_paddeds = pad_sequences(test_seqences, maxlen=max_length, 
                             truncating=trunc_type)

# 딥러닝 
import tensorflow as tf

model = tf.keras.Sequential([
  tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(6, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

num_epochs = 5

model.fit(imdb_paddeds, train_labels, epochs=num_epochs,
         validation_data=(test_paddeds, test_labels))