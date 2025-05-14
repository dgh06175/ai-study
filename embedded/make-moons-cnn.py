import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# x에 값이 두개이므로 (자로좌표, 세로좌표) x를 둘로 나눠야함
# 다중 선형 회귀는 Make-moons 에 어울리지 않음, 가로 좌표 값이 여러개일때 사용하는게 다중 선형 회귀
# 다항 회귀로 모델링.
# 1. 데이터 생성
X, _ = make_moons(n_samples=500, noise=0.1, random_state=0)
# print(f"{X[0]:.4f}, {X[1]:.4f}")
# print(X)
x1 = X[:, 0].reshape(-1, 1)
x2 = X[:, 1].reshape(-1, 1)

# 2. 훈련/테스트 분리
x1_train, x1_test, x2_train, x2_test = train_test_split(x1, x2, test_size=0.2, random_state=42)

# 3. 다항 특성 생성 (예: 5차까지 확장)
poly = PolynomialFeatures(degree=6)
x1_train_poly = poly.fit_transform(x1_train)
x1_test_poly = poly.transform(x1_test)

# 4. 다항 회귀 모델 학습
model = LinearRegression()
model.fit(x1_train_poly, x2_train)

# 5. 예측 및 평가
x2_pred = model.predict(x1_test_poly)
mse = mean_squared_error(x2_test, x2_pred)
print(f"다항 회귀 평균제곱오차 (MSE): {mse:.4f}")

# 6. 예측 결과 시각화
x1_range = np.linspace(x1.min(), x1.max(), 300).reshape(-1, 1)
x1_range_poly = poly.transform(x1_range)
x2_range_pred = model.predict(x1_range_poly)

plt.figure(figsize=(8, 5))
plt.scatter(x1, x2, label='Actual', alpha=0.4)
plt.plot(x1_range, x2_range_pred, color='red', label='Predict (deg=5)', linewidth=2)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('x1 vs x2 Prediction')
plt.legend()
plt.grid(True)
plt.show()
