import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

results = {}

def calculate_exponential_average(df,ghis,alphas,train_size):
	for alpha in alphas:
	    fit = SimpleExpSmoothing(ghis).fit(smoothing_level=alpha, optimized=False)
	    df['Simple Exponential Smoothing'] = fit.fittedvalues
	    train_size_index = int(len(df) * train_size)
	    train_data = df[:train_size_index]
	    test_data = df[train_size_index:]
	    y_true = test_data['GHI']
	    y_pred = test_data['Simple Exponential Smoothing']
	    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
	    r2 = r2_score(y_true, y_pred)
	    mae = mean_absolute_error(y_true, y_pred)
	    results[alpha] = {'RMSE': rmse, 'R2': r2, 'MAE': mae, 'Forecast': y_pred}
	print('Alpha\t\tRMSE\t\tR2\t\tMAE')
	for alpha, result in results.items():
	    print(f'{alpha:.1f}\t\t{result["RMSE"]:.2f}\t\t{result["R2"]:.2f}\t\t{result["MAE"]:.2f}')
	best_alpha = max(results, key=lambda x: results[x]['R2'])
	print(f'\nBest alpha: {best_alpha}')
	plot(test_data,alphas,y_pred)

def plot(test_data,alphas,y_pred):
	plt.figure(figsize=(10,6))
	plt.plot(test_data.index[-100:], test_data['GHI'].tail(100), label='Actual Values')
	colors = plt.cm.plasma(np.linspace(0,1,len(alphas)))
	for i, alpha in enumerate(alphas):
	    y_pred = results[alpha]['Forecast'].tail(100)
	    plt.plot(test_data.index[-100:], y_pred, label=f'Alpha={alpha:.1f}',color=colors[i], alpha=0.8)
	plt.xlabel('Observations')
	plt.ylabel('GHI')
	plt.title('Actual vs. Predicted Values for Different Alphas (Last 100 Observations)')
	plt.legend()
	plt.xlim(test_data.index[-100], test_data.index[-1])
	plt.show()
