import numpy as np
import pandas as pd
from arch.unitroot import KPSS
from matplotlib import pyplot as plt

store_sales = pd.read_csv('../data/ACCL_Sales.csv')
tsdf = store_sales[['Delivery Qty', 'Delivery Date']]


store_sales['year'] = pd.DatetimeIndex(store_sales['Delivery Date']).year
store_sales['month'] = pd.DatetimeIndex(store_sales['Delivery Date']).month

store_sales['date'] = store_sales['year'].astype(str) + '-' + store_sales['month'].astype(str) + '-01'
store_sales = store_sales.groupby(['date']).sum().reset_index()

tsdf['dates'] = store_sales['date']
tsdf['sales_index'] = store_sales['Delivery Qty']
tsdf = tsdf[['sales_index', 'dates']]
tsdf['log_sales_index'] = np.log(tsdf.sales_index)

for i in range(3):
    tsdf[f'sales_index_lag_{i + 1}'] = tsdf.sales_index.shift(i + 1)
    tsdf.dropna(inplace=True)
    tsdf[f'log_sales_index_lag_{i + 1}'] = np.log(tsdf[f'sales_index_lag_{i + 1}'])
    tsdf[f'log_difference_{i + 1}'] = tsdf.log_sales_index - tsdf[f'log_sales_index_lag_{i + 1}']

kpss_test = KPSS(tsdf.log_difference_1)
print(kpss_test.summary().as_text())
print(len(tsdf.sales_index), len(tsdf.log_difference_1))
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(tsdf['dates'],tsdf.sales_index)
plt.title('O')
plt.subplot(1, 2, 2)
plt.plot(tsdf['dates'], tsdf.log_difference_1)
plt.title('L')
plt.show()