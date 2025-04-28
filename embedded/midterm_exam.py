from sklearn.datasets import make_moons
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

X, y = make_moons(n_samples=300, noise=0.1, random_state=42)

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

mse = mean_squared_error(y, pred)
R2 = r2_score(y, pred)

print('MSE: ', mse)
print('R2: ', R2)
