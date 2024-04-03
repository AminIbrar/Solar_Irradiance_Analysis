import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.ar_model import AutoReg

results = {}

def calculate_auto_regression(df,ghis,lags,train_size):
	for lag in lags:
	    model = AutoReg(ghis, lags=lag)
	    fit = model.fit()
	    train_size_index = int(len(df) * train_size)
	    train_data = df[:train_size_index]
	    test_data = df[train_size_index:]
	    y_true = test_data['GHI']
	    y_pred = fit.predict(start=train_size_index, end=len(df)-1)
	    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
	    r2 = r2_score(y_true, y_pred)
	    mae = mean_absolute_error(y_true, y_pred)
	    results[lag] = {'RMSE': rmse, 'R2': r2, 'MAE': mae, 'Forecast': y_pred}
	print('Lag\t\tRMSE\t\tR2\t\tMAE')
	for lag, result in results.items():
	    print(f'{lag}\t\t{result["RMSE"]:.2f}\t\t{result["R2"]:.2f}\t\t{result["MAE"]:2f}')
	plot(test_data,lags)


def plot(test_data,lags):
	plt.figure(figsize=(10,6))
	plt.plot(test_data.index[-100:], test_data['GHI'].tail(100), label='Actual Values')
	colors = plt.cm.jet(np.linspace(0,1,len(lags)))   
	       
	for i, lag in enumerate(lags):
	    y_pred = results[lag]['Forecast'][-100:]
	    plt.plot(test_data.index[-100:], y_pred, label=f'Lag={lag}', color=colors[i])
	plt.xlabel('Observations')
	plt.ylabel('GHI')
	plt.title('Actual vs. Predicted Values for Different Lags (Last 100 Observations)')
	plt.legend()
	plt.show()
