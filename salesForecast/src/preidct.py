import mlflow
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logged_model = 'runs:/f07132f3eda94521ac5babf7f876ef4f/model'

store_sales = pd.read_csv('../data/train.csv')
store_sales = store_sales.drop(['store', 'item'], axis=1)
store_sales['date'] = pd.to_datetime(store_sales['date'])
#
store_sales['date'] = store_sales['date'].dt.to_period("M")
monthly_sales = store_sales.groupby('date').sum().reset_index()
monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()
monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
monthly_sales = monthly_sales.dropna()

supervised_data = monthly_sales.drop(['date', 'sales'], axis=1)

for i in range(1, 13):
    col_name = 'month_' + str(i)
    supervised_data[col_name] = supervised_data['sales_diff'].shift(i);

supervised_data = supervised_data.dropna().reset_index(drop=True)
train_data = supervised_data[:-12]
test_data = supervised_data[-12:]
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_data)
test_data = scaler.transform(test_data)
X_test, y_test = test_data[:, 1:], test_data[:, 0:1]
#
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)
# # Predict on a Pandas DataFrame.
#
#print(X_test)
print(y_test,loaded_model.predict(X_test))

