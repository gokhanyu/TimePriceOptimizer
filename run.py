from core.fixinterrupthandler import FixInterruptHandling
from core.options import Options
from core.datareader import DataReader
from core.plotter import Plotter
from core.dataprocessor import DataProcessor
from core.datapreprocessor import DataPreProcessor
from core.dataprepare import DataPrepare



import math
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy import array
from matplotlib.dates import (drange)
from datetime import (datetime, timedelta)



#

FixInterruptHandling().fix()

#

if (Options.PrepareFeatures) :
  dataPrepared = DataPrepare.run(Options.OutputFeaturesPath, Options.InputFeaturesPath, Options.TrainDataStartDate, Options.FuturePredictionStartDate, 
                                 Options.FuturePredictionEndDate())

#

output_data_dict_train, output_data_dict_future = DataReader.get_output_features()

#

input_data_dict_train, input_data_dict_future = DataReader.get_input_features()

#
Options.InputFeatureSize = len(input_data_dict_train.values())

#


plotter = Plotter()
#plotter.plot_date_price(date_values, price_values, Options.AssetName)


wave_no = 0
input1 = list(input_data_dict_train.values())[wave_no] 
#plotter.plot_xy(date_values, input1, list(input_data_dict.keys())[wave_no], 'Dates', 'Power')



# Construct PDS

input_pds = pd.DataFrame.from_dict(input_data_dict_train)
real_close_pds = pd.DataFrame(data=output_data_dict_train[Options.RealClosePricesKey])
output_pds = pd.DataFrame(data=output_data_dict_train[Options.OutputValuesKey])
date_values = output_data_dict_train[Options.OutputDateValuesKey]

future_input_pds = pd.DataFrame.from_dict(input_data_dict_future)
future_output_pds, future_real_close_pds, future_date_values = DataReader.get_future_output_features(output_data_dict_future)

# End Construct PDS



# Data Pre Processor

pre_processor = DataPreProcessor()
#output_pds = pre_processor.to_wavelet(output_pds)
#output_pds = pre_processor.filter_by_savgol(output_pds, 51, 3)


processor = DataProcessor(input_pds, output_pds, date_values, Options.TrainDataSize, Options.TestDataSize)



# LAZY LOAD Neural Network libs
from core.nnmodel import NNModel



### ONE BY ONE START
if (Options.DataWindow == Options.DataWindowType.OneByOne) :

  x_train_o, y_train_o, date_train = processor.get_one_by_one_data('train', seq_len=1, multiply_y_vector = Options.MultiplyDataByCustomFactor, normalise=Options.NormaliseData)
  x_val_o, y_val_o, date_val = processor.get_one_by_one_data('val', seq_len=1, multiply_y_vector = Options.MultiplyDataByCustomFactor, normalise=Options.NormaliseData)
  model = NNModel()

  model.build_one_by_one_model()
  callbacks = model.get_callbacks(Options.KerasOneByOneEpochs)
  model.fit_one_by_one(x_train_o, y_train_o, x_val_o, y_val_o, Options.KerasOneByOneBatchSize, Options.KerasOneByOneEpochs, callbacks)
  y_train_o_pred = model.predict(x_train_o)

  
  plotter.plot_different_scale(y_train_o_pred, y_train_o, date_train, y_label1 = "Train Prediction", y_label2 = "Train")

  plotter.plot_different_scale(model.predict(x_val_o), y_val_o, date_val, y_label1 = "Validation Prediction", y_label2 = "Validation")

  processor2 = DataProcessor(input_pds, real_close_pds, date_values, Options.TrainDataSize, Options.TestDataSize) #DELETE THIS
  temp, y_test_real, date_real = processor2.get_one_by_one_data('test', seq_len=1, multiply_y_vector = Options.MultiplyDataByCustomFactor, normalise=Options.NormaliseData) #DELETE THIS

  x_test_o, y_test_o, date_test = processor.get_one_by_one_data('test', seq_len=1, multiply_y_vector = Options.MultiplyDataByCustomFactor, normalise=Options.NormaliseData)
  y_test_o_pred = model.predict(x_test_o)

  plotter.plot_different_scale(y_test_o_pred, y_test_real, y_label1 = "Test Prediction", y_label2 = "Real")
  plotter.plot_different_scale(y_test_o_pred, y_test_o, date_real, y_label1 = "Test Prediction", y_label2 = "Real RPO")
  plotter.plot_different_scale(model.predict(future_input_pds), future_real_close_pds, np.array(future_date_values), y_label1 = "Future Prediction", y_label2 = "Future Real")

### ONE BY ONE END

  

if (Options.DataWindow == Options.DataWindowType.OneByOneTelosSearch) :
  from core.nntelossearch import NNTelosSearch

  telosSearch = NNTelosSearch()
  x_train_o, y_train_o = processor.get_train_data_OBO(seq_len=1, normalise=Options.NormaliseData)
  telosSearch.minimize(x_train_o, y_train_o, [], [])

#TELOS END



if (Options.DataWindow == Options.DataWindowType.WindowBatch) :

  #x_train, y_train = processor.get_train_data(seq_len=Options.WindowSequenceLength, normalise=Options.NormaliseData)
  #y_train = y_train * Options.MultiplyDataByCustomFactor
  #x_test, y_test = processor.get_test_data(seq_len = Options.WindowSequenceLength, normalise = Options.NormaliseData)
  #y_test = y_test * Options.MultiplyDataByCustomFactor

  model = NNModel()
  model.build_windowed_batch_model()
  callbacks = model.get_callbacks(Options.KerasWindowedEpochs)

  x_train, y_train, date_train = processor.get_window_train_data(arr_type='train', seq_len = Options.WindowSequenceLength, 
                                                     step = Options.WindowShiftStep, feature_len = Options.InputFeatureSize,
                                                     multiply_y_vector = 1)

  x_val, y_val, date_val = processor.get_window_train_data(arr_type='val', seq_len = Options.WindowSequenceLength, 
                                                     step = Options.WindowShiftStep, feature_len = Options.InputFeatureSize,
                                                     multiply_y_vector = 1)

  x_test, y_test, date_test = processor.get_window_train_data(arr_type='test', seq_len = Options.WindowSequenceLength, 
                                                     step = Options.WindowShiftStep, feature_len = Options.InputFeatureSize,
                                                     multiply_y_vector = 1)

  if (Options.FlattenWindowVector) :
    x_train = np.asarray(x_train, dtype= np.float32).reshape(-1, Options.WindowSequenceLength*Options.InputFeatureSize)
    x_val = np.asarray(x_val, dtype= np.float32).reshape(-1, Options.WindowSequenceLength*Options.InputFeatureSize)
    x_test = np.asarray(x_test, dtype= np.float32).reshape(-1, Options.WindowSequenceLength*Options.InputFeatureSize)


  model.fit_one_by_one(x_train, y_train, x_val, y_val, Options.KerasWindowedBatchSize, Options.KerasWindowedEpochs, callbacks)


  if (Options.FlattenWindowVector) :
    plotter.plot_different_scale(model.predict(x_train), y_train, date_train, y_label1 = "Train Prediction", y_label2 = "Train")
    plotter.plot_different_scale(model.predict(x_val), y_val, date_val, y_label1 = "Validation Prediction", y_label2 = "Validation")
    plotter.plot_different_scale(model.predict(x_test), y_test, date_test, y_label1 = "Test Prediction", y_label2 = "Test")
  else:
    train_pred_point_by_point = model.predict_point_by_point(x_train)
    plotter.plot_different_scale(train_pred_point_by_point, y_train, date_train, y_label1 = "Train Prediction", y_label2 = "Train")

    test_pred_point_by_point = model.predict_point_by_point(x_test)
    plotter.plot_different_scale(test_pred_point_by_point, y_test, date_test, y_label1 = "Test Prediction", y_label2 = "Real")

#WINDOW_END

print("THE_END")


  
