import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Generate data (trend + seasonality + noise)
np.random.seed(42)
n = 120
time = np.arange(n)
seasonal = 10 * np.sin(2 * np.pi * time / 12)
trend = 0.3 * time
noise = np.random.normal(0, 2, n)
data = 20 + trend + seasonal + noise
ts = pd.Series(data)

# Apply ARIMA (first-order differencing is needed)
model = ARIMA(ts, order=(2, 1, 2))  # Simple example
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=12)
plt.figure(figsize=(12, 4))
plt.plot(ts, label="Observed")
plt.plot(np.arange(len(ts), len(ts)+12), forecast, label="Forecast", color="red")
plt.title("ARIMA Forecast on Trend + Seasonal Data")
plt.legend()
plt.grid()
plt.show()
