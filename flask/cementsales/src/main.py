import pickle
import pandas as pd
import numpy as np

scaler = pickle.load(open('../model/scaler.save', 'rb'))
model = pickle.load(open('../model/model.pkl', 'rb'))
from flask import Flask, request

app = Flask(__name__)


@app.route('/sales_predict/', methods=['GET', 'POST'])
def sales_predict():
    data = request.json
    test = pd.DataFrame(data)
    A = model.predict(test)
    B = np.reshape(A, (1, -1))
    B = scaler.inverse_transform(B)
    B = B.flatten()
    df = pd.DataFrame(B)
    return df.to_json()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
