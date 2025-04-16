import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error

np.random.seed(42)
x = np.linspace(-1, 1, 100).reshape(-1, 1)
y = 2 * x + 3 + 0.5 * np.random.randn(100, 1)

x_train = x[:20]
y_train = y[:20]

x_test = x
y_test = y

model_no_dropout = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(1,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])
model_no_dropout.compile(optimizer='adam', loss='mse')
model_no_dropout.fit(x_train, y_train, epochs=300, verbose=0)

pred_no_dropout = model_no_dropout.predict(x_test)
mse_no_dropout = mean_squared_error(y_test, pred_no_dropout)

# 모델 2: 드롭아웃 있음
model_with_dropout = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(1,)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1)
])
model_with_dropout.compile(optimizer='adam', loss='mse')
model_with_dropout.fit(x_train, y_train, epochs=300, verbose=0)

pred_with_dropout = model_with_dropout.predict(x_test)
mse_with_dropout = mean_squared_error(y_test, pred_with_dropout)

print("MSE (No Dropout): {:.3f}".format(mse_no_dropout))
print("MSE (Dropout): {:.3f}".format(mse_with_dropout))