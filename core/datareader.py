from core.options import Options

import csv
import time
import pandas as pd
import os
import numpy as np
from datetime import (datetime, timedelta)



class DataReader :
    
    @staticmethod
    def read_dataframe(filename, tab_char = '\t'):
      pds = pd.read_csv(filename, tab_char, header=0, infer_datetime_format=True, nrows=Options.ReadNumberOfRowsFromFile)
      row_count = len(pds)
      return pds, row_count
    

    @staticmethod
    def get_files_in_dir(dir_path, filter):
      files = []

      for file in os.listdir(dir_path):
        if file.endswith(filter):
          files.append(os.path.join(dir_path, file))

      return files


    @staticmethod
    def read_data(data_path, store_dates) :
      output_pds = DataReader.read_dataframe(data_path, ',')
      output_shape = output_pds[1]
      values = np.array(output_pds[0].iloc[:,1]) #can be price close values or even an oscillator values
      
      date_values = []

      if (store_dates) :
        date_strings = np.array(output_pds[0].iloc[:,0])

        for date_str in date_strings : 
          date_values.append(datetime.strptime(date_str, '%Y.%m.%d %H:%M'))

      return values, date_values


    @staticmethod
    def get_dictionary_from_csv(path, store_dates):
      
      files = DataReader.get_files_in_dir(path, ".csv")
      file_dict = {}

      for file in files:
        values, dates = DataReader.read_data(file, store_dates)
        filename, file_extension = os.path.splitext(os.path.basename(file))
        
        file_dict[filename] = values

        if (store_dates) :
          file_dict[filename+"_date"] = dates

      return file_dict


    @staticmethod
    def get_output_features() :

      output_data_dict_train = DataReader.get_dictionary_from_csv(Options.OutputReadyTrainAllPath, True)
      output_data_dict_future = DataReader.get_dictionary_from_csv(Options.OutputReadyFuturesPath, True)

      return output_data_dict_train, output_data_dict_future


    @staticmethod
    def get_input_features() :

      input_data_dict_train = DataReader.get_dictionary_from_csv(Options.InputReadyTrainAllPath, False)
      input_data_dict_future = DataReader.get_dictionary_from_csv(Options.InputReadyFuturesPath, False)

      return input_data_dict_train, input_data_dict_future


    @staticmethod
    def get_future_output_features(output_data_dict_future) :

      files = DataReader.get_files_in_dir(Options.InputReadyFuturesPath, ".csv")

      future_real_close_pds = pd.DataFrame(data=output_data_dict_future[Options.RealClosePricesKey])

      output_values = output_data_dict_future[Options.OutputValuesKey]
      out_real_values = output_data_dict_future[Options.RealClosePricesKey]
      out_date_values = output_data_dict_future[Options.OutputDateValuesKey]

      output_value_dic = {}
      output_real_value_dic = {}

      for i in range(len(out_date_values)) :
        output_value_dic[out_date_values[i]] =  output_values[i]
        output_real_value_dic[out_date_values[i]] =  out_real_values[i]

      #tmp values will be 
      for file in files :
        tmp_values, future_input_dates = DataReader.read_data(file, True)
        break

      future_real_close = [0]*len(tmp_values)

      for i in range(len(tmp_values)) :
        dic_lookup_day = future_input_dates[i]

        if (dic_lookup_day in output_value_dic) :
          tmp_values[i] = output_value_dic[dic_lookup_day]
          future_real_close[i] = output_real_value_dic[dic_lookup_day]
        else :
          if (i > 0) :
            tmp_values[i] = tmp_values[i-1]
            future_real_close[i] = future_real_close[i-1]
          
      for i in range(len(tmp_values) - 1, -1, -1) :
        if (((i+1) < len(tmp_values)) and tmp_values[i] == 0):
          tmp_values[i] = tmp_values[i+1]
          future_real_close[i] = future_real_close[i+1]



      future_output_pds = pd.DataFrame(data=tmp_values)
      future_real_close_pds = pd.DataFrame(data=future_real_close)
      return future_output_pds, future_real_close_pds, future_input_dates





