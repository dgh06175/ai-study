import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(0)
x_train = np.random.rand(500, 3)
y_train = 3 * x_train[:, 0] - 2 * x_train[:, 1] + 1 * x_train[:, 2] + 5
y_train = y_train.reshape(-1, 1)

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(3,), name = "Hidden_Layer_1"),
    layers.Dense(64, activation='relu', name = "Hidden_Layer_2"),
    layers.Dense(1, name = "Output_Layer")
])

print("모델 구조:")
print(model.summary())

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(x_train, y_train, epochs=100, batch_size=16, verbose=1)

x_test = np.array([[0.5, 0.2, 0.1], [1.9, 0.0, 0.5]])
y_pred = model.predict(x_test)

print("\n입력값: ")
print(x_test)
print("\n예측 결과: ")
print(y_pred.flatten())
