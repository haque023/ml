import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
store_sales = pd.read_csv('../data/train.csv')
store_sales = store_sales.drop(['store', 'item'], axis=1)
store_sales['date'] = pd.to_datetime(store_sales['date'])
print(store_sales.info())
