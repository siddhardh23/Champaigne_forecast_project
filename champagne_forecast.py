import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import DateOffset

# Load and clean data
df = pd.read_csv('perrin-freres-monthly-champagne-.csv')
df.columns = ['Month', 'Sales']
df = df[df['Month'].str.contains(r'\d{4}-\d{2}', na=False)]
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df = df.asfreq('MS')
df['Sales'] = df['Sales'].interpolate()

# Fit SARIMA model
model = SARIMAX(df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Forecast 24 months
future_dates = [df.index[-1] + DateOffset(months=x) for x in range(1, 25)]
forecast = results.get_forecast(steps=24)
forecast_df = pd.DataFrame({'forecast': forecast.predicted_mean}, index=future_dates)

# Plot and save
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Sales'], label='Actual Sales', color='blue')
plt.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', linestyle='--', color='orange')
plt.title('Champagne Sales Forecast (SARIMA)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('champagne_forecast.png')
