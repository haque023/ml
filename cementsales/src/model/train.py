import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


async def get_test_datas():
    store_sales = pd.read_csv('../data/ACCL_Sales.csv')
    plt.figure(figsize=(15, 5))
    plt.xlabel("Date")
    plt.ylabel("sales")
    plt.title("Monthly Customer Sales Differance")

    store_sales['year'] = pd.DatetimeIndex(store_sales['Delivery Date']).year
    store_sales['month'] = pd.DatetimeIndex(store_sales['Delivery Date']).month
    store_sales['date'] = store_sales['year'].astype(str) + '-' + store_sales['month'].astype(str) + '-01'

    store_sales['time'] = pd.to_datetime(store_sales['date']).astype('int64') / 10 ** 9
    store_sales = store_sales[['time', 'Delivery Qty']]
    store_sales = store_sales.groupby(['time']).sum().reset_index()

    store_sales = store_sales.sort_values(['time'])
    Q3 = store_sales['Delivery Qty'].quantile(0.80)
    store_sales = store_sales.where(store_sales['Delivery Qty'] <= Q3)
    store_sales['Quantity'] = store_sales['Delivery Qty']
    store_sales['Date'] = store_sales['time']
    scaler = MinMaxScaler(feature_range=(1, 2))
    store_sales = store_sales.drop(columns='Date')
    store_sales = store_sales.dropna().reset_index(drop=True)
    df = pd.DataFrame(store_sales['Quantity'])
    store_salesd = scaler.fit_transform(df.to_numpy())
    store_sales['ScaleQuantity'] = store_salesd.flatten().tolist()

    store_sales = store_sales.drop(columns='Quantity')
    store_sales = store_sales.drop(columns='Delivery Qty')
    store_sales = store_sales.dropna()

    X = store_sales.drop(columns=['ScaleQuantity'])
    Y = store_sales['ScaleQuantity']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}


async def train_cement_data():
    tData = await get_test_datas()
    lr = LinearRegression()
    lr.fit(tData['x_train'], tData['y_train'])
    return lr


def get_metrics(y_train, y_predict):
    # print(y_train, y_predict)
    # from sklearn.metrics import accuracy_score
    # acc = accuracy_score(y_train, y_predict)
    return {'accuracy': 0}
