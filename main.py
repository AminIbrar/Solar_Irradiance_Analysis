import pandas as pd 
import numpy as np
from moving_average import calculate_moving_average
from exponential_average import calculate_exponential_average
from double_exponential_average import calculate_double_exponential_average
from auto_regression import calculate_auto_regression
from ARIMA import calculate_arima

def load_data():
	df = pd.read_csv('data.csv', skiprows=2)
	df = df[df['Month']>9]
	data = df[df['GHI'].notnull()]
	ghis = data['GHI'].values
	return data, ghis

def main():
	window_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
	alphas = np.linspace(0.1, 1, 10)
	betas = np.linspace(0.1, 1, 10)
	lags = [1,4,6,20,30,40,350,1000]
	train_size = 0.90

	data,ghis = load_data()

	print('\t\t\t\t Moving Average ')

	calculate_moving_average(data, ghis, window_sizes,train_size)

	# print('\t\t\t\t Exponential Moving Average ')

	# calculate_exponential_average(data, ghis, alphas,train_size)

	# print('\t\t\t\t Double Exponential Moving Average ')

	# calculate_double_exponential_average(data, ghis, alphas,betas,train_size)

	# print('\t\t\t\t Auto Regression ')

	# calculate_auto_regression(data,ghis,lags,train_size)

	# print('\t\t\t\t ARIMA ')

	# calculate_arima(data,train_size)


if __name__ == "__main__":
    main()
