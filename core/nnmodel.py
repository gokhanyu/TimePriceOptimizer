from core.options import Options
from core.utils import Timer
from core.signalstopping import SignalStopping
from core.utils import Utils


import os
import signal
import math
import numpy as np
import datetime as dt
import keras as keras
import pandas as pd


from numpy import array
from numpy import newaxis
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, LSTM, GaussianNoise, BatchNormalization, LeakyReLU
from keras.layers import Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.constraints import max_norm
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.utils.generic_utils import get_custom_objects
from keras.utils import np_utils
import tensorflow as tf
import keras.backend as K
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))




class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)

#K.clear_session()
#get_custom_objects().update({'swish': Activation(swish)})
#activation = 'swish'






class NNModel():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self):
		self.model = Sequential()
		self.model_loss = ''

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath)

	def purge_saved_models(self) :
		saved_files = os.listdir(Options.KerasNNSaveDirectory)
		for item in saved_files :
			if item.endswith(".hdf5") :
				os.remove(os.path.join(Options.KerasNNSaveDirectory, item))


	def get_callbacks(self, epochs):
		csv_logger = CSVLogger('keras.nn.training.log', append=True)
		
		callbacks = [csv_logger]

		#self.purge_saved_models()

		#new file for each epoch
		filepath = os.path.join(Options.KerasNNSaveDirectory, '%s-e%s.hdf5' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S'), str(epochs)))

		if (Options.KerasModelCheckpointSave) : 
			checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=Options.KerasVerbose, save_weights_only=False, save_best_only=Options.KerasModelCheckpointSaveOnlyBestModel, mode='max', period = Options.KerasCheckPointForNEpoch)
			callbacks.append(checkpoint)

		if (Options.KerasEarlyStopping) : 
			early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=Options.KerasEarlyStoppingPatience, verbose=Options.KerasVerbose, mode='auto')
			callbacks.append(early_stop)

		if (Options.KerasReduceLearningRateOnPlateu) :
			reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=2, min_lr=0.000001, verbose=Options.KerasVerbose)
			callbacks.append(reduce_lr)
		
		if (Options.KerasSignalStopping) :
			signalstop = SignalStopping(doubleSignalExits = True, verbose=Options.KerasVerbose)
			callbacks.append(signalstop)

		return callbacks



	def build_one_by_one_model_ngut(self) : 
		timer = Timer()
		timer.start()
		Options.KerasEarlyStopping = False
		self.model.add(Dense(64, activation='sigmoid', input_dim=Options.InputFeatureSize, activity_regularizer=keras.regularizers.l2(0.01))) 
		self.model.add(BatchNormalization()) 
		self.model.add(LeakyReLU()) 
		self.model.add(Dense(16, activity_regularizer=keras.regularizers.l2(0.01))) 
		self.model.add(BatchNormalization()) 
		self.model.add(LeakyReLU()) 
		self.model.add(Dense(1)) 
		self.model.add(Activation('linear'))
		optimizer = RMSprop(lr = 0.00050)
		self.model_loss = 'mae' #'mean_squared_error', 'mae'
		self.model.compile(optimizer = optimizer, loss = self.model_loss, metrics = list(set(['mae', self.model_loss])))
		print('[Model] Model Compiled')
		print('[Model] Model Loss Function is %s.' % (self.model_loss))
		timer.stop()
		self.model.summary()
		return self.model



	def build_one_by_one_model_gut(self) :
		timer = Timer()
		timer.start()
		Options.KerasEarlyStopping = False
		self.model.add(Dropout(0.2, input_shape=(Options.InputFeatureSize,)))
		self.model.add(Dense(70, input_dim = Options.InputFeatureSize, activation = 'sigmoid', 
        kernel_initializer='glorot_uniform', bias_initializer='zeros')) #CAREFUL SIGMOID , NO NORMAL INIT
		self.model.add(Dropout(0.2))
		#self.model.add(Dense(25, activation = 'relu'))
		self.model.add(Dense(1, activation = 'linear')) #CAREFUL SIGMOID
		optimizer = RMSprop(lr = 0.00050)
		#optimizer = RMSprop()
		#optimizer = SGD()
		#optimizer = Adam()
		self.model_loss = 'mae' #'mean_squared_error', 'mae', 'logcosh'
		self.model.compile(optimizer = optimizer, loss = self.model_loss, metrics = list(set(['mae', 'mean_absolute_percentage_error', 'mean_squared_error', 'logcosh', self.model_loss])))
		print('[Model] Model Compiled')
		print('[Model] Model Loss Function is %s.' % (self.model_loss))
		timer.stop()
		self.model.summary()
		return self.model



	def build_one_by_one_model_test(self) :
		timer = Timer()
		timer.start()
		Options.KerasEarlyStopping = False
		self.model.add(Dropout(0.2, input_shape=(Options.InputFeatureSize,)))
		self.model.add(Dense(60, input_dim = Options.InputFeatureSize, activation = 'sigmoid', 
        kernel_initializer='glorot_uniform', bias_initializer='zeros')) #CAREFUL SIGMOID , NO NORMAL INIT
		self.model.add(Dropout(0.2))
		#self.model.add(Dense(25, activation = 'relu'))
		self.model.add(Dense(1, activation = 'linear')) #CAREFUL SIGMOID
		optimizer = RMSprop(lr = 0.00050)
		#optimizer = RMSprop()
		#optimizer = SGD()
		#optimizer = Adam()
		self.model_loss = 'mae' #'mean_squared_error', 'mae', 'logcosh'
		self.model.compile(optimizer = optimizer, loss = self.model_loss, metrics = list(set(['mae', 'mean_absolute_percentage_error', 'mean_squared_error', 'logcosh', self.model_loss])))
		print('[Model] Model Compiled')
		print('[Model] Model Loss Function is %s.' % (self.model_loss))
		timer.stop()
		self.model.summary()
		return self.model


  #Change the model from here
	def build_one_by_one_model(self) :
		return self.build_one_by_one_model_test()



	def predict(self, x_input, name) :

		if (len(x_input) == 0) :
			print("SKIPPING predicting %s, length 0." % (name))
			return []

		y_pred = self.model.predict(x_input)
		return y_pred


	def save_pred_to_csv(self, pred_dates, pred_y, save_dirs_array, fname) :
		if (len(pred_y) > 0) :
		  for save_dir in save_dirs_array :
		    Utils.ensure_directory(save_dir)
		    save_fname = os.path.join(save_dir, fname)

		    df = pd.DataFrame()
		    df['Date'] = pred_dates
		    
		    if (len(pred_y) < len(pred_dates)):
		      new_pred_y = np.zeros((len(pred_dates), 1))
		      new_pred_y[0:len(pred_y)] = pred_y
		      df['Prediction'] = new_pred_y
		    else:
		      df['Prediction'] = pred_y

		    export_csv = df.to_csv(save_fname, index=None, header=True)


	def fit_one_by_one(self, x_train_o, y_train_o, x_val_o, y_val_o, batch_size, epochs, callbacks) :
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		print('[Model] Train Shape: %s' % str(x_train_o.shape))
		print('[Model] Validation Shape: %s' % str(x_val_o.shape))

		try:
			self.model.fit(
					x_train_o, y_train_o, 
					epochs = epochs, batch_size = batch_size, 
					validation_data = [x_val_o, y_val_o],
					verbose = Options.KerasVerbose, callbacks = callbacks)		
		except KeyboardInterrupt:
			print('Keyboard interrupt.')
			pass

		save_fname = os.path.join(Options.KerasNNSaveDirectory, '%s-e%s.hdf5' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S'), str(len(self.model.history.epoch))))
		self.model.save(save_fname)
		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()




	def build_windowed_batch_model(self):
		timer = Timer()
		timer.start()
		Options.FlattenWindowVector = True
		Options.WindowSequenceLength = 3
		Options.WindowShiftStep = 1
		Options.KerasEpochs = 30
		Options.KerasWindowedBatchSize = 5
		Options.KerasEarlyStopping = False

		self.model.add(Dropout(0.3, input_shape=(Options.WindowSequenceLength*Options.InputFeatureSize,)))
		self.model.add(Dense(150, input_dim = Options.InputFeatureSize, activation = 'sigmoid', 
        kernel_initializer='glorot_uniform', bias_initializer='zeros')) #CAREFUL SIGMOID , NO NORMAL INIT
		self.model.add(Dropout(0.3))
		self.model.add(Dense(32, activation = 'relu'))
		self.model.add(Dense(1, activation = 'linear')) #CAREFUL SIGMOID
		optimizer = RMSprop(lr = 0.00050)
		#optimizer = RMSprop()
		#optimizer = SGD()
		#optimizer = Adam()
		self.model_loss = 'mae' #'mean_squared_error', 'mae'
		self.model.compile(optimizer = optimizer, loss = self.model_loss, metrics = list(set(['mae', self.model_loss])))
		print('[Model] Model Compiled')
		print('[Model] Model Loss Function is %s.' % (self.model_loss))
		timer.stop()
		self.model.summary()
		return self.model


	def build_windowed_batch_simple_model(self):
		timer = Timer()
		timer.start()
		Options.FlattenWindowVector = True
		Options.WindowSequenceLength = 3
		Options.WindowShiftStep = 1
		Options.KerasEpochs = 30
		Options.KerasWindowedBatchSize = 5
		Options.KerasEarlyStopping = False

		self.model.add(Dropout(0.3, input_shape=(Options.WindowSequenceLength*Options.InputFeatureSize,)))
		self.model.add(Dense(150, input_dim = Options.InputFeatureSize, activation = 'sigmoid', 
        kernel_initializer='glorot_uniform', bias_initializer='zeros')) #CAREFUL SIGMOID , NO NORMAL INIT
		self.model.add(Dropout(0.3))
		#self.model.add(Dense(64, activation = 'relu'))
		self.model.add(Dense(1, activation = 'linear')) #CAREFUL SIGMOID
		optimizer = RMSprop(lr = 0.00050)
		#optimizer = RMSprop()
		#optimizer = SGD()
		#optimizer = Adam()
		self.model_loss = 'mae' #'mean_squared_error', 'mae'
		self.model.compile(optimizer = optimizer, loss = self.model_loss, metrics = list(set(['mae', self.model_loss])))
		print('[Model] Model Compiled')
		print('[Model] Model Loss Function is %s.' % (self.model_loss))
		timer.stop()
		self.model.summary()
		return self.model

  #used train_generator to train for 2 epochs. todo
	def build_windowed_batch_LSTM_model(self):
		timer = Timer()
		timer.start()
		Options.WindowSequenceLength = 56
		Options.WindowShiftStep = 1
		Options.KerasEpochs = 30
		Options.KerasWindowedBatchSize = 400
		Options.KerasEarlyStopping = False

		#self.model.add(Reshape((Options.WindowSequenceLength, Options.InputFeatureSize), input_shape=(Options.WindowSequenceLength, Options.InputFeatureSize,)))
		self.model.add(LSTM(100, input_shape=(Options.WindowSequenceLength, Options.InputFeatureSize), return_sequences = True))
		self.model.add(Dropout(0.2))
		self.model.add(LSTM(100, return_sequences = True))
		self.model.add(LSTM(100, return_sequences = False))
		self.model.add(Dropout(0.2))
		self.model.add(Dense(1, kernel_initializer = 'normal', activation = 'linear'))
		optimizer = RMSprop(lr = 0.00050)
		#optimizer = Adam()
		self.model_loss = 'mean_squared_error' #'mean_squared_error', 'mae'
		self.model.compile(optimizer = optimizer, loss = self.model_loss, metrics = list(set(['mae', self.model_loss])))
		print('[Model] Model Compiled')
		timer.stop()
		return self.model


	def build_windowed_batch_CONV_model(self):
		timer = Timer()
		timer.start()
		Options.KerasEpochs = 20
		Options.KerasWindowedBatchSize = 400

		#self.model.add(Reshape((Options.WindowSequenceLength, Options.InputFeatureSize), input_shape=(input_shape,)))
		self.model.add(Conv1D(100, 10, activation='relu', input_shape=(Options.WindowSequenceLength, Options.InputFeatureSize)))
		self.model.add(Conv1D(100, 10, activation='relu'))
		self.model.add(MaxPooling1D(3))
		self.model.add(Conv1D(160, 10, activation='relu'))
		#self.model.add(Conv1D(160, 10, activation='relu'))
		self.model.add(GlobalAveragePooling1D())
		self.model.add(Dropout(0.5))
		self.model.add(Dense(1, activation = 'linear'))

		optimizer = RMSprop(lr = 0.00050)
		#optimizer = Adam()
		self.model_loss = 'mae' #'mean_squared_error', 'mae'
		self.model.compile(optimizer = optimizer, loss = self.model_loss, metrics = list(set(['mae', self.model_loss])))
		print('[Model] Model Compiled')
		timer.stop()



	#WINDOWED BATCH TRAIN METHOD
	def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
		timer = Timer()
		timer.start()
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size, %s batches per epoch' % (epochs, batch_size, steps_per_epoch))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%Y%m%d-%H%M%S'), str(epochs)))

		callbacks = [
			#ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True),
			SignalStopping(doubleSignalExits = True, verbose=Options.KerasVerbose)
		]

		self.model.fit_generator(
			data_gen,
			#validation_data = [],
			steps_per_epoch=steps_per_epoch,
			epochs=epochs,
			callbacks=callbacks,
			#,workers=1
		)
		
		print('[Model] Training Completed. Model saved as %s' % save_fname)
		timer.stop()



	def predict_point_by_point(self, data):
		#Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
		print('[Model] Predicting Point-by-Point...')
		predicted = self.model.predict(data)
		predicted = np.reshape(predicted, (predicted.size,))
		return predicted



	def predict_sequences_multiple(self, data, window_size, prediction_len):
		#Predict sequence of 50 steps before shifting prediction run forward by 50 steps
		print('[Model] Predicting Sequences Multiple...')
		prediction_seqs = []
		for i in range(int(len(data)/prediction_len)):
			curr_frame = data[i*prediction_len]
			predicted = []
			for j in range(prediction_len):
				predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
				curr_frame = curr_frame[1:]
				curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
			prediction_seqs.append(predicted)
		return prediction_seqs



	def predict_sequence_full(self, data, window_size):
		#Shift the window by 1 new prediction each time, re-run predictions on new window
		print('[Model] Predicting Sequences Full...')
		curr_frame = data[0]
		predicted = []
		for i in range(len(data)):
			predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
			curr_frame = curr_frame[1:]
			curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
		return predicted
