import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.utils.np_utils import to_categorical

iris = load_iris()
X, y = iris.data, np.array(to_categorical(iris.target), dtype=np.int32)
X, y = shuffle(X, y)
to_categorical(y)

train_x = X[:120,]
test_x = X[120:,]
train_y = y[:120,]
test_y = y[120:,]

model = keras.Sequential()
model.add(keras.layers.Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer = keras.optimizers.SGD(learning_rate=0.04),
              metrics=['accuracy'])
model.summary()

hist= model.fit(train_x, train_y, epochs=200, batch_size=10, shuffle=True)

plt.plot(hist.history['loss'])
plt.plot(hist.history['accuracy'])
plt.show()

prob = model.predict(test_x)
pred = prob.argmax(axis=-1)

print("테스트 샘플 예측 결과:")
for i in range(len(pred)):
    print(f"샘플 {i+1}: 실제 = {iris.target_names[pred[i]]}, 예측 = {iris.target_names[list(test_y[i]).index(1)]}")
