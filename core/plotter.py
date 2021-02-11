import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import (datetime, timedelta)
from matplotlib.dates import (drange)
from matplotlib.dates import num2date #, bytespdate2num
from matplotlib.ticker import Formatter
from core.options import Options


class MyFormatter(Formatter):
  def __init__(self, dates, fmt='%Y-%m-%d'):
      self.dates = dates
      self.fmt = fmt

  def __call__(self, x, pos=0):
      'Return the label for time x at position pos'
      ind = int(np.round(x))
      if ind >= len(self.dates) or ind < 0:
          return ''

      return num2date(self.dates[ind]).strftime(self.fmt)


class Plotter :
  
  # For plotting input data
  def plot_xy(self, date_values, close_values, title='', xLabel = '', yLabel = '') : 
    if (Options.PlottingEnabled == False) :
      return

    # Matplotlib prefers datetime instead of np.datetime64.
    #date = date_values_obj.astype('O')
    fig, ax = plt.subplots()
    ax.plot(date_values, close_values)
    fig.autofmt_xdate()

    ax.fmt_xdata = mdates.DateFormatter('%Y.%m.%d %H:%M')
    ax.set_title(title, fontsize=18)
    plt.xlabel(xLabel,fontsize=18)
    plt.ylabel(yLabel,fontsize=18)
    plt.show()


  # For drawing stock market close price graph
  def plot_date_price(self, date_values, close_values, title='', xLabel = '', yLabel = '') : 
    self.plot_xy(date_values, close_values, title, 'Date', 'Close Price')


  #WRITE PARAMETER TYPES
  def plot_different_scale(self, data1, data2, common_x_axis = np.array([]), x_label1 = "x", y_label1 = "y", y_label2 = "y") : 
    if (len(data1) == 0 and len(data2) == 0) :
      return

    if (Options.PlottingEnabled == False) :
      return
    

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel(x_label1)
    ax1.set_ylabel(y_label1, color=color)
    
    if (common_x_axis.size == 0) :
      ax1.plot(data1, color=color)
    else :
      ax1.plot(common_x_axis, data1, color=color)
    
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(y_label2, color=color)  # we already handled the x-label with ax1

    if (common_x_axis.size == 0) :
      ax2.plot(data2, color=color)
    else :
      ax2.plot(common_x_axis, data2, color=color)

    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    #plt.gca().xaxis.set_major_locator(mdates.HourLocator(byhour=[0,1]))
    #plt.gcf().autofmt_xdate()
    plt.show()


  def plot_results_multiple(self, predicted_data, true_data, prediction_len):
      fig = plt.figure(facecolor='white')
      ax = fig.add_subplot(111)
      ax.plot(true_data, label='True Data')

      # Pad the list of predictions to shift it in the graph to it's correct start
      for i, data in enumerate(predicted_data):
          padding = [None for p in range(i * prediction_len)]
          plt.plot(padding + data, label='Prediction')
          plt.legend()
      plt.show()