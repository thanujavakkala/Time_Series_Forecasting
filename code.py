import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Generate synthetic time series data
np.random.seed(0)
n = 100
time = np.arange(n)
data = np.random.randn(n)  # Replace with your actual time series data

# Create a Pandas DataFrame
df = pd.DataFrame({'Time': time, 'Value': data})

# Visualize the data
plt.plot(df['Time'], df['Value'])
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Original Time Series Data')
plt.show()

# Fit ARIMA model
model = ARIMA(df['Value'], order=(5,1,0))  # Replace with appropriate order (p, d, q)
results = model.fit()

# Forecast future values
n_forecast = 10  # Number of periods to forecast
forecast, stderr, conf_int = results.forecast(steps=n_forecast)

# Visualize the forecast
plt.plot(df['Time'], df['Value'], label='Original Data')
plt.plot(np.arange(n, n + n_forecast), forecast, label='Forecasted Data')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.title('Time Series Forecasting using ARIMA')
plt.show()

# Evaluate the model
mse = mean_squared_error(df['Value'][1:], results.fittedvalues)
print('Mean Squared Error:', mse)