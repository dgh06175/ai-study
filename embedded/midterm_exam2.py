import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# 데이터 로드 및 셔플
iris = load_iris()
X, y = iris.data, iris.target
X, y = shuffle(X, y, random_state=42)

# 원-핫 인코딩
y = to_categorical(y)

# 훈련/테스트 분할
train_x = X[:120,]
test_x = X[120:,]
train_y = y[:120,]
test_y = y[120:,]

# 모델 구성 (input_shape=(4,) 명시)
model = keras.Sequential([
    layers.Dense(10, activation='relu', input_shape=(4,)),
    layers.Dense(3, activation='softmax')
])

# 모델 컴파일
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.SGD(learning_rate=0.04),
              metrics=['accuracy'])

# 모델 구조 출력
model.summary()

# 모델 학습
hist = model.fit(train_x, train_y, epochs=200, batch_size=10, shuffle=True, verbose=0)

# 학습 과정 시각화
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()

# 테스트 데이터 예측
prob = model.predict(test_x)
pred = prob.argmax(axis=-1)

# 예측 결과 출력
print("테스트 샘플 예측 결과:")
for i in range(len(pred)):
    print(f"샘플 {i+1}: 실제 = {iris.target_names[list(test_y[i]).index(1)]}, 예측 = {iris.target_names[pred[i]]}")
