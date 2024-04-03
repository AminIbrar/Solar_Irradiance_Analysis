
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


results = {}
def calculate_moving_average(df, ghis,window_sizes,train_size):
	for window_size in window_sizes:
		moving_avg = np.nan*np.zeros(len(ghis))
		for i in range(len(ghis)):
			start_idx = max(0, i - window_size + 1)
			moving_avg[i] = np.mean(ghis[start_idx:i+1]) 
		train_size_index = int(len(ghis)*train_size)
		train_data = ghis[:train_size_index]
		test_data = ghis[train_size_index:]
		y_true = test_data
		y_pred = moving_avg[train_size_index:]

		rmse = np.sqrt(mean_squared_error(y_true,y_pred))
		r2 = r2_score(y_true,y_pred)
		mae = mean_squared_error(y_true,y_pred)

		results[window_size] = {'RMSE': rmse, 'R2': r2, 'MAE': mae}

	print('Window Size\tRMSE\t\tR2\t\tMAE')
	for window_size, result in results.items():
		print(f'{window_size}\t\t{result["RMSE"]:.2f}\t\t{result["R2"]:.2f}\t\t{result["MAE"]:.2f}')
	best_window_size = max(results, key=lambda x: results[x]['R2'])
	print(f'\nBest window size: {best_window_size}')
	plot(df,ghis,window_sizes)




def plot(df,ghis,window_sizes):
	plt.figure(figsize=(10,6))
	plt.plot(df.index, ghis, label='Actual Values', color='black')
	colors = plt.cm.plasma(np.linspace(0,1,len(window_sizes)))
	for i, window_size in enumerate(window_sizes):
		moving_avg = np.nan * np.zeros(len(ghis))
		for j in range(window_size, len(ghis)):
			moving_avg[j] = np.mean(ghis[j-window_size:j])
		plt.plot(df.index, moving_avg, label=f'Window Size {window_size}', color=colors[i], alpha=0.8)
	plt.xlabel('Observations')
	plt.ylabel('GHI')
	plt.title('Actual vs. Predicted Values for Different Window Sizes')
	plt.xlim(df.index[-500], df.index[-1])
	plt.ylim(0,700)
	plt.legend()

	plt.show()
