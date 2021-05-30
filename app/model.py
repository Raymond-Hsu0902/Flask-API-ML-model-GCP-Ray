# -*- coding: UTF-8 -*-
import pickle
import gzip
import joblib

# 載入Model
#with gzip.open('app/model/xgboost-iris.pgz', 'r') as f:
    #xgboostModel = pickle.load(f)
xgboostModel = joblib.load("app/model/xgb1.joblib.dat")

def predict(input):
    pred=xgboostModel.predict(input)[0]
    print(pred)
    return pred
