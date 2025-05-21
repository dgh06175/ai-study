import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.utils import to_categorical

# 문자 매핑
char_to_idx = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz")}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# 데이터 만들기
seq_length = 3
data = []
text = "abcdefghijklmnopqrstuvwxyz"
for i in range(len(text) - seq_length):
    input_seq = text[i:i+seq_length]
    target_char = text[i+seq_length]
    data.append((input_seq, target_char))

# 벡터화
x = np.zeros((len(data), seq_length, len(char_to_idx)))
y = np.zeros((len(data), len(char_to_idx)))
for i, (seq, target) in enumerate(data):
    for t, char in enumerate(seq):
        x[i, t, char_to_idx[char]] = 1
    y[i, char_to_idx[target]] = 1

# 모델 정의 (SimpleRNN)
model = Sequential()
model.add(SimpleRNN(16, input_shape=(seq_length, len(char_to_idx))))
model.add(Dense(len(char_to_idx), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=200, verbose=0)

# 예측 함수
def predict(seq):
    input_data = np.zeros((1, seq_length, len(char_to_idx)))
    for t, char in enumerate(seq):
        input_data[0, t, char_to_idx[char]] = 1
    prediction = model.predict(input_data, verbose=0)
    predicted_idx = np.argmax(prediction)
    return idx_to_char[predicted_idx]

test_input = "abc"
print(f"입력: {test_input} → 예측 결과: {predict(test_input)}")
