import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

np.random.seed(42)
x1 = 2 * np.random.random((100, 1)) - 1
x2 = 2 * np.random.random((100, 1)) - 1
y = 4 + 3 * x1 + 2 * x2 + np.random.normal(0, 1, (100, 1))

model = LinearRegression()
x = np.hstack((x1, x2))
print(x.shape, y.shape)

model.fit(x, y)

print("회귀 개수 (기울기): ", model.coef_)
print("절편: ", model.intercept_[0])

x_text = np.array([[1, 2], [2, 3]])
y_pred = model.predict(x_text)

mse = mean_squared_error(y, model.predict(x))
print("Mean Squared Error (MSE): ", mse)

mae = mean_absolute_error(y, model.predict(x))
print("Mean Absolute Error (MAE): ", mae)