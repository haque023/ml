import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow import MlflowClient
import mlflow.sklearn
from pprint import pprint


def fetch_logged_data(run_id):
    client = MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
    return data.params, data.metrics, tags, artifacts


mlflow.sklearn.autolog()


store_sales = pd.read_csv('../data/train.csv')
store_sales = store_sales.drop(['store', 'item'], axis=1)
store_sales['date'] = pd.to_datetime(store_sales['date'])

store_sales['date'] = store_sales['date'].dt.to_period("M")
monthly_sales = store_sales.groupby('date').sum().reset_index()
monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
monthly_sales = monthly_sales.dropna()

supervised_data = monthly_sales.drop(['date', 'sales'], axis=1)
for i in range(1, 13):
    col_name = 'month_' + str(i)
    supervised_data[col_name] = supervised_data['sales_diff'].shift(i)

supervised_data = supervised_data.dropna().reset_index(drop=True)
print(supervised_data)
train_data = supervised_data[:-12]
test_data = supervised_data[-12:]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

X_train, y_train = train_data[:, 1:], train_data[:, 0:1]
X_test, y_test = test_data[:, 1:], test_data[:, 0:1]

y_train = y_train.ravel()
y_test = y_test.ravel()
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

sales_dates = monthly_sales['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)
act_sales = monthly_sales['sales'][-13:].to_list()
print(act_sales)

lr_model = LinearRegression()
with mlflow.start_run() as run:
    lr_model.fit(X_train, y_train)
    lr_pre = lr_model.predict(X_test)
    lr_pre = lr_pre.reshape(-1, 1)
    lr_pre_test_set = np.concatenate([lr_pre, X_test], axis=1)
    lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)
    result_list = []
    for index in range(0, len(lr_pre_test_set)):
        result_list.append((lr_pre_test_set[index][0])+act_sales[index])
    lr_pre_series = pd.Series(result_list, name="Linear Prediction")
    predict_df = predict_df.merge(lr_pre_series, left_index=True, right_index=True)
    lr_mse = np.sqrt(mean_squared_error((predict_df['Linear Prediction']), monthly_sales['sales'][-12:]))
    lr_mae = np.sqrt(mean_absolute_error((predict_df['Linear Prediction']), monthly_sales['sales'][-12:]))
    lr_r2 = r2_score((predict_df['Linear Prediction']), monthly_sales['sales'][-12:])

    print("mse", lr_mse)
    print("mae", lr_mae)
    print("mae", lr_r2)

    # plt.figure(figsize=(15, 5))
    # plt.plot(monthly_sales['date'], monthly_sales['sales'])
    # plt.plot(predict_df['date'], predict_df['Linear Prediction'])
    # plt.xlabel("Date")
    # plt.ylabel("sales")
    # plt.title("Monthly Customer Sales Differance")
    # plt.legend(['Actual Sales', 'predicted sales'])
    # plt.show()
params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)
pprint(params)
pprint(metrics)
pprint(tags)
pprint(artifacts)

# prediction = pd.DataFrame(X_test, columns=['predictions']).to_csv('prediction.csv')

