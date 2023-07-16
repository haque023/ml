from pprint import pprint

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from mlflow import MlflowClient
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mlflow_exp_id = '0'
def fetch_logged_data(run_id):
    client = MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts

mlflow.set_tracking_uri("http://10.209.99.12:5000")
mlflow.sklearn.autolog()
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
# X = X.sort_values(['time'])

with mlflow.start_run(experiment_id = mlflow_exp_id) as run:
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    i = lr.intercept_
    c = lr.coef_
    print(x_test)
    import pickle

    model = pickle.load(open('model/model.pkl', 'rb'))
    pickle.dump(model, open('model/model.pkl', 'wb'))
    model = pickle.load(open('model/model.pkl', 'rb'))
    print(scaler.inverse_transform(model.predict(x_test)))
    # signature = infer_signature(x_test, lr.predict(x_test))
    # mlflow.sklearn.log_model(lr, "sales", signature=signature)
    # x_test = x_test.sort_values(by=['time'])
    #
    # y_predict = lr.predict(x_test)
    #
    # store_sales['Datetime'] = pd.to_datetime(store_sales['time'], unit='s')
    # x_test['Datetime'] = pd.to_datetime(x_test['time'], unit='s')
    # x_train['Datetime'] = pd.to_datetime(x_train['time'], unit='s')
    #
    # plt.plot(store_sales['Datetime'], store_sales['ScaleQuantity'])
    # plt.plot(x_test['Datetime'], y_predict)
    # plt.show()
    # lr_mse = np.sqrt(mean_squared_error(y_predict, y_test))
    # lr_mae = np.sqrt(mean_absolute_error(y_predict, y_test))
    # lr_r2 = r2_score(y_predict, y_test)
    #
    # print(lr_r2)
params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
# pprint(params)
# pprint(metrics)
# pprint(tags)
# pprint(artifacts)