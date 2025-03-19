import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 3, 5, 6, 8])

model = LinearRegression()

model.fit(x, y)

y_pred = model.predict(x)

print("예측 값: ", y_pred)
print("회귀 개수 (기울기): ", model.coef_)
print("절편: ", model.intercept_)