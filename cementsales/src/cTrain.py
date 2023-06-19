import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from mlflow import MlflowClient
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def fetch_logged_data(run_id):
    client = MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts

mlflow.sklearn.autolog()

store_sales = pd.read_csv('../data/salesdata.csv')
store_sales=store_sales.dropna()
plt.figure(figsize=(15, 5))
plt.xlabel("Date")
plt.ylabel("sales")
plt.title("Monthly Customer Sales Differance")
store_sales['time'] = pd.to_datetime(store_sales['Date']).astype('int64') / 10**9
Q3 = store_sales['Quantity'].quantile(0.95)
store_sales = store_sales.where(store_sales['Quantity']<=Q3)
store_sales = store_sales.drop(columns='Date')
store_sales = store_sales.dropna().reset_index(drop=True)
scaler = MinMaxScaler(feature_range=(1, 2))
df = pd.DataFrame(store_sales['Quantity'])
store_salesd = scaler.fit_transform(df.to_numpy())
store_sales['ScaleQuantity'] = store_salesd.flatten().tolist()
# store_sales['diff'] = store_sales['ScaleQuantity'].diff()
store_sales = store_sales.drop(columns='Quantity')
store_sales = store_sales.dropna()
#
X = store_sales.drop(columns=['ScaleQuantity'])
Y = store_sales['ScaleQuantity']

with mlflow.start_run() as run:


    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=10)
    #
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    #
    i = lr.intercept_
    c = lr.coef_
    #
    #
    plt.plot(store_sales['time'], store_sales['ScaleQuantity'])
    signature = infer_signature(x_test, lr.predict(x_test))
    mlflow.sklearn.log_model(lr, "sales", signature=signature)

    x_test = x_test.sort_values(by=['time'])
    y_predict = lr.predict(x_test)
    plt.plot(x_test['time'], y_predict)
    plt.show()
    print(r2_score(y_test, y_predict))
