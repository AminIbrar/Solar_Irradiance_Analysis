import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm



def calculate_arima(df,train_size):
	result = adfuller(df['GHI'])
	model = sm.tsa.ARIMA(df['GHI'], order=(3,1,1))
	fit = model.fit()
	# Make predictions
	train_size_index = int(len(df) * train_size) # Change this value to 0.98
	train_data = df[:train_size_index]
	test_data = df[train_size_index:]
	y_true = test_data['GHI']
	y_pred = fit.predict(start=test_data.index[0], end=test_data.index[-1])
	# Evaluate model performance
	rmse = np.sqrt(mean_squared_error(y_true, y_pred))
	r2 = r2_score(y_true, y_pred)
	mae = mean_absolute_error(y_true, y_pred)
	print('RMSE: {:.2f}'.format(rmse))
	print('R2 Score: {:.2f}'.format(r2))
	print('MAE: {:.2f}'.format(mae))

	plot(test_data,y_pred)

def plot(test_data,y_pred):
	plt.figure(figsize=(10, 6))
	plt.plot(test_data.index[-100:], test_data['GHI'].tail(100), label='Actual Values')
	y_pred_last_100 = y_pred[-100:]
	plt.plot(test_data.index[-100:], y_pred_last_100, label='ARIMA(3,1,1)', color='red')
	plt.xlabel('Observations')
	plt.ylabel('GHI')
	plt.title('Actual vs. Predicted Values (ARIMA) for Last 100 Observations')
	plt.legend()
	plt.show()