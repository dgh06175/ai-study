import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([[1], [11], [21], [31], [41]])
y = np.array([100, 67, 43, 21, 0])

model = LinearRegression()

model.fit(x, y)

y_pred = model.predict(x)

print(y_pred)
print(model.coef_)