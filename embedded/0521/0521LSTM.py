import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. 학습용 문장
text = """
machine learning is a method of data analysis that automates analytical model building
using algorithms that iteratively learn from data machine learning allows computers
to find hidden insights without being explicitly programmed
"""

# 2. 단어 토크나이즈
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
word_index = tokenizer.word_index
total_words = len(word_index) + 1

# 3. 시퀀스 데이터 생성
tokens = tokenizer.texts_to_sequences([text])[0]
input_sequences = []
for i in range(2, len(tokens)):
    input_sequences.append(tokens[i-2:i+1])
input_sequences = np.array(input_sequences)
x = input_sequences[:, :-1]
y = to_categorical(input_sequences[:, -1], num_classes=total_words)

# 4. 모델 구성
model = Sequential()
model.add(Embedding(input_dim=total_words, output_dim=10, input_length=2))
model.add(LSTM(64))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 모델 학습
model.fit(x, y, epochs=300, verbose=0)

# 6. 예측: "machine learning" 다음 단어
seed_text = "machine learning"
seed_seq = tokenizer.texts_to_sequences([seed_text])[0]
seed_seq = pad_sequences([seed_seq], maxlen=2)
predicted_idx = np.argmax(model.predict(seed_seq, verbose=0))
for word, idx in tokenizer.word_index.items():
    if idx == predicted_idx:
        print(f"입력: '{seed_text}' → 예측된 다음 단어: '{word}'")
        break
