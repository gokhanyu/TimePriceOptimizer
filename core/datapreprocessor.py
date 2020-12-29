from core.options import Options
from core.datareader import DataReader

import numpy as np
import pandas as pd
import pywt  ##TO INSTALL => pip install pywavelets [DO NOT TYPE "pip install pywt"]
import matplotlib.pyplot as plt


class DataPreProcessor:
    def __init__(self):
        return

    def init(self, split, feature_split):
        self.split = split
        self.feature_split = feature_split
        self.output_pds = DataReader.read_dataframe(Options.OutputFeaturesPath, ',')
        self.stock_data = output_pds[0]


    #https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
    #Savitzky - Golay filter
    def filter_by_savgol(self, output_pds, window_size = 51, poly_order = 3) :
      from scipy.signal import savgol_filter
      output = output_pds.values
      x = np.reshape(output, (output.size,)).astype(float)
      yhat = savgol_filter(x, window_size, poly_order) # window size 51, polynomial order 3
      out_pd = pd.DataFrame(yhat)
      return out_pd
