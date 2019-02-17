import math
import numpy as np
import pandas as pd

from numpy import array


class DataProcessor():
	"""A class for loading and transforming data (np array, pandas conversions)"""

	def __init__(self, input_pds, output_pds, date_values, train_split, test_split):
		assert (len(input_pds) == len(output_pds))

		train_idx = int(len(input_pds) * train_split)
		validate_idx = int(len(input_pds) * (1 - test_split))

		self.date_values = date_values

		self.data_train_x = input_pds.values[:train_idx]
		self.data_train_y = output_pds.values[:train_idx]
		self.date_train = np.array(date_values[:train_idx])

		self.data_val_x = input_pds.values[train_idx : validate_idx]
		self.data_val_y = output_pds.values[train_idx : validate_idx]
		self.date_val = np.array(date_values[train_idx : validate_idx])

		self.data_test_x = input_pds.values[validate_idx:]
		self.data_test_y = output_pds.values[validate_idx:]
		self.date_test = np.array(date_values[validate_idx:])

		#self.data_train_x = dataframe.get(cols).values[:i_split]
		#self.data_test = dataframe.get(cols).values[i_split:]

		self.len_train = len(self.data_train_x)
		self.len_val = len(self.data_val_x)
		self.len_test = len(self.data_test_x)
		self.len_train_windows = None


  
	#One By One
	def get_one_by_one_data(self, arr_type, seq_len, multiply_y_vector, normalise):

		if (arr_type == 'train') :
			x_train_o = self.data_train_x #[:-1]  #WAS SHIFTING WORKS BETTER?
			y_train_o = self.data_train_y #[1:]
			return x_train_o, y_train_o, self.date_train
		elif (arr_type == 'val') :
			x_val_o = self.data_val_x #[:-1]
			y_val_o = self.data_val_y #[1:]
			return x_val_o, y_val_o, self.date_val
		elif (arr_type == 'test') :
			x_test_o = self.data_test_x #[:-1]
			y_test_o = self.data_test_y #[1:]
			return x_test_o, y_test_o, self.date_test
		else:
		  ERROR



	# Number of steps to advance in each iteration (for me, it should always	
	# be equal to the time_steps in order to have no overlap between segments)
	# step = time_steps
	def get_window_train_data(self, arr_type, seq_len, step, feature_len, multiply_y_vector) : 
		data_windows_x = []
		data_windows_y = []

		X = None
		Y = None
		len = 0
		dates_range = np.array([])
		dates_source = np.array([])

		if (arr_type == 'train') :
		  X = self.data_train_x
		  Y = self.data_train_y
		  len = self.len_train
		  dates_source = self.date_train
		elif (arr_type == 'val') :
		  X = self.data_val_x
		  Y = self.data_val_y
		  len = self.len_val
		  dates_source = self.date_val
		elif (arr_type == 'test') :
		  X = self.data_test_x
		  Y = self.data_test_y
		  len = self.len_test
		  dates_source = self.date_test
		else:
		  ERROR

		for i in range(0, len - seq_len, step):
			if (i + seq_len < len) :
			  data_windows_x.append(X[i : i + seq_len])
			  data_windows_y.append(Y[i + seq_len - 1])
			  np.append(dates_range, dates_source[i])

		reshaped_data = np.asarray(data_windows_x, dtype= np.float32).reshape(-1, seq_len, feature_len)

		data_windows_y = data_windows_y * multiply_y_vector 

		data_windows_y = np.array(data_windows_y)
		reshaped_data = np.array(reshaped_data)

		return reshaped_data, data_windows_y, dates_range



	#Window Batch
	def get_train_data(self, seq_len, normalise):
		'''
		Create x, y train data windows
		Warning: batch method, not generative, make sure you have enough memory to
		load data, otherwise use generate_training_window() method.
		'''
		data_x = []
		data_y = []

		for i in range(self.len_train - seq_len):
			x, y = self._next_window(i, seq_len, normalise)
			data_x.append(x)
			data_y.append(y)
		return np.array(data_x), np.array(data_y)


	#Window Batch
	def get_test_data(self, seq_len, normalise):
		'''
		Create x, y test data windows
		Warning: batch method, not generative, make sure you have enough memory to
		load data, otherwise reduce size of the training split.
		'''
		data_windows_x = []
		data_windows_y = []
		for i in range(self.len_test - seq_len):
			data_windows_x.append(self.data_test_x[i : i + seq_len])
			data_windows_y.append(self.data_test_y[i + seq_len - 1])

		#data_windows = np.array(data_windows).astype(float)
		#data_windows = self.normalise_windows(data_windows,
		#single_window=False) if normalise else data_windows

		x = np.array(data_windows_x).astype(float)
		y = np.array(data_windows_y).astype(float)
		return x,y

	def _next_window(self, i, seq_len, normalise):
		'''Generates the next data window from the given index location i'''
		window = self.data_train_x[i:i + seq_len]
		window = self.normalise_windows(window, single_window=True)[0] if normalise else window
		x = window
		y = self.data_train_y[i + seq_len - 1]
		return x, y

	def ok_to_normalise(self, data) :
		if (type(data) is float) :
			return True
		else :
			return False

	def normalise_windows(self, window_data, single_window=False):
		'''Normalise window with a base value of zero'''
		normalised_data = []
		window_data = [window_data] if single_window else window_data

		for window in window_data:
			normalised_window = []

			for col_i in range(window.shape[1]):

				if (self.ok_to_normalise(window[0, col_i])) :
				  normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
				  normalised_window.append(normalised_col)
				else : 
				  aa = 5

			normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
			normalised_data.append(normalised_window)
		return np.array(normalised_data)

	def generate_train_batch(self, seq_len, batch_size, normalise):
		'''Yield a generator of training data from filename on given list of cols split for train/test'''
		i = 0

		while i < (self.len_train - seq_len):
			x_batch = []
			y_batch = []

			for b in range(batch_size):
				if i >= (self.len_train - seq_len):
					# stop-condition for a smaller final batch if data doesn't
					# divide evenly
					yield np.array(x_batch), np.array(y_batch)
					i = 0
				x, y = self._next_window(i, seq_len, normalise)
				x_batch.append(x)
				y_batch.append(y)
				i += 1
			yield np.array(x_batch), np.array(y_batch)


	