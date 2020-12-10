# Author: T. Lovelock

#---------------------
# NOTES
# Imports data from yahoo finance.
# Only the raw data is imported and exported here, this is because the data needs to be scaled and transformed back
# and it is more efficent to do this in predictionmodel.py to save the scaling values.
#---------------------

from pandas_datareader import data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sb

#Obtain Bond Yield Data
start_date = '2019-11-01'; end_date = '2020-10-31'
bond_yield = pd.DataFrame(data.DataReader('^TNX', 'yahoo', start_date, end_date)['Adj Close'], columns=['Adj Close'])

fig, ax = plt.subplots()
sb.set_style("darkgrid")
sb.lineplot(data=bond_yield, x=bond_yield.index, y='Adj Close')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
ax.xaxis.set_major_locator(plt.MaxNLocator(9))
plt.ylabel('Yield %', size=12)
plt.xlabel('Date', size=12)
plt.title('US 10 Year Bond Yield', size=12)
plt.savefig('10 Yr Bond Yield')
plt.close()

#Obtain S&P 500 Data and Nasdaq Data
start_date = '2020-01-02'; end_date = '2020-10-31'
SP_500_Data = pd.DataFrame(data.DataReader('^GSPC', 'yahoo', start_date, end_date)['Adj Close'], columns=['Adj Close'])
Comp_Data = pd.DataFrame(data.DataReader('^IXIC', 'yahoo', start_date, end_date)['Adj Close'], columns=['Adj Close'])

SP_500_Data['Adj Close'] = SP_500_Data['Adj Close']/SP_500_Data['Adj Close'][0]
Comp_Data['Adj Close'] = Comp_Data['Adj Close']/Comp_Data['Adj Close'][0]

fig, ax = plt.subplots()
sb.set_style("darkgrid")
sb.lineplot(data=SP_500_Data, x=SP_500_Data.index, y='Adj Close', legend='brief', label="S&P 500")
sb.lineplot(data=Comp_Data, x=Comp_Data.index, y='Adj Close', legend='brief', label="Nasdaq")
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
ax.xaxis.set_major_locator(plt.MaxNLocator(9))
plt.ylabel('Base - 2 Jan 2020', size=12)
plt.xlabel('Date', size=12)
plt.title('Performance Comparison', size=12)
plt.savefig('Performance Comparison')
plt.close()

#Pre-Covid 19 Data
start_date = '2019-03-01'; end_date = '2019-12-31'
pre_covid = data.DataReader('AMZN', 'yahoo', start_date, end_date)['Adj Close']

#Post-Covid 19 Data
start_date = '2020-01-02'; end_date = '2020-10-30'
post_covid = data.DataReader('AMZN', 'yahoo', start_date, end_date)['Adj Close']

#Plot data to check for errors
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.plot(pre_covid.values); ax1.title.set_text('Pre Covid'); ax1.set_xlabel('Days'); ax1.set_ylabel('Adj Price')
ax2.plot(post_covid.values); ax2.title.set_text('Post Covid'); ax2.set_xlabel('Days')
plt.savefig('Amazon')

#Dataset for testing (will be viewed when making predictions)
start_date = '2015-01-02'; end_date = '2015-05-29'
predict_set = data.DataReader('AMZN', 'yahoo', start_date, end_date)['Adj Close']

#Export Data
outres = open('data1.csv', 'w'); np.savetxt(outres, pre_covid); outres.close()
outres = open('data2.csv', 'w'); np.savetxt(outres, post_covid); outres.close()
outres = open('predict.csv', 'w'); np.savetxt(outres, predict_set); outres.close()
