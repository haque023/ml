import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import math
from sklearn import metrics
from sklearn.metrics import confusion_matrix


plt.rc("font",size=14)
df=pd.read_csv('../data/banking.csv')
print(df.head())
print(df.columns)