import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 1. 정상 시계열 + 이상값 삽입
np.random.seed(42)
n = 120
x = np.arange(n)
data = 10 + np.sin(2 * np.pi * x / 24) + np.random.normal(0, 0.5, n)
anomalies = [30, 60, 90]
data[anomalies] += [5, -6, 7]

# 2. ARIMA 학습
model = ARIMA(data, order=(2, 0, 0))
model_fit = model.fit()
fitted = model_fit.fittedvalues
residuals = data - fitted

# 3. 허용 범위 계산 (잔차 기반)
residual_std = np.std(residuals)
upper_bound = fitted + 3 * residual_std
lower_bound = fitted - 3 * residual_std

# 4. 이상값 탐지
anomaly_indices = np.where((data > upper_bound) | (data < lower_bound))[0]

# 5. 시각화
plt.figure(figsize=(12, 5))
plt.plot(data, label='Observed', color='black')
plt.plot(fitted, label='Fitted', linestyle='--', color='blue')
plt.fill_between(np.arange(n), lower_bound, upper_bound, color='skyblue', alpha=0.3, label='±3σ Bound')
plt.scatter(anomaly_indices, data[anomaly_indices], color='red', label='Detected Anomalies', zorder=5)
plt.title("Anomaly Detection with ARIMA (±3σ Bounds)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# 이상값 출력
print("Detected Anomaly Indices:", anomaly_indices)
