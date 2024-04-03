import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.holtwinters import Holt

results = {}

def calculate_double_exponential_average(df,ghis,alphas,betas,train_size):
	for alpha in alphas:
	    for beta in betas:
	        fit = Holt(ghis).fit(smoothing_level=alpha, smoothing_slope=beta)
	        df['Double Exponential Smoothing'] = fit.fittedvalues
	        train_size_index = int(len(df) * train_size)
	        train_data = df[:train_size_index]
	        test_data = df[train_size_index:]
	        y_true = test_data['GHI']
	        y_pred = test_data['Double Exponential Smoothing']
	        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
	        r2 = r2_score(y_true, y_pred)
	        mae = mean_absolute_error(y_true, y_pred)
	        results[(alpha, beta)] = {'RMSE': rmse, 'R2': r2, 'MAE': mae, 'Forecast': y_pred}
	print('Alpha\t\tBeta\t\tRMSE\t\tR2\t\tMAE')
	for alpha, beta in results.keys():
	    result = results[(alpha, beta)]
	    print(f'{alpha:.1f}\t\t{beta:.1f}\t\t{result["RMSE"]:.2f}\t\t{result["R2"]:.2f}\t\t{result["MAE"]:.2f}')
	best_alpha, best_beta = max(results, key=lambda x: results[x]['R2'])
	print(f'\nBest alpha: {best_alpha}, Best beta: {best_beta}')

	plot(test_data,alphas,betas,results)

def plot(test_data,alphas,betas,results):
	# Plot actual values and predicted values for different alphas and betas for last 1
	plt.figure(figsize=(10,6))
	plt.plot(test_data.index[-100:], test_data['GHI'].tail(100), label='Actual Values')
	alphas, betas = zip(*results.keys())
	colors = plt.cm.jet(np.linspace(0,1,len(alphas)))
	for i, (alpha, beta) in enumerate(results.keys()):
	    y_pred = results[(alpha, beta)]['Forecast'].tail(100)
	    plt.plot(test_data.index[-100:], y_pred, label=f'Alpha={alpha:.1f}, Beta={beta:.1f}')
	plt.xlabel('Observations')
	plt.ylabel('GHI')
	plt.title('Actual vs. Predicted Values for Different Alphas and Betas (Last 100 Observations)')
	plt.legend()
	plt.xlim(test_data.index[-100], test_data.index[-1])
	plt.show()