# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : model_lgb.py
# Time       ：2022/10/27 13:53
# Author     ：author name
# version    ：python 3.6
# Description：
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('train_data.csv')


# 离线数据训练模型
def model_lgb(offline_data):
    train_x = offline_data.iloc[:, :75]
    train_y = offline_data['cluster']

    print(train_x.head())

    x, test_x, y, test_y = train_test_split(
        train_x,
        train_y,
        test_size=0.05,
        random_state=1,
        stratify=train_y  ## 这里保证分割后y的比例分布与原数据一致
    )

    print(test_x.head())
    lgb_train = lgb.Dataset(x, y)
    params = {'num_leaves': 60,
              'min_data_in_leaf': 30,
              'objective': 'multiclass',
              'num_class': 33,
              'max_depth': -1,
              'learning_rate': 0.03,
              "min_sum_hessian_in_leaf": 6,
              "boosting": "gbdt",
              "feature_fraction": 0.9,
              "bagging_freq": 1,
              "bagging_fraction": 0.8,
              "bagging_seed": 11,
              "lambda_l1": 0.1,
              "verbosity": -1,
              "nthread": 15,
              'metric': 'multi_logloss',
              "random_state": 2019,
              # 'device': 'gpu'
              }
    gbm = lgb.train(params, lgb_train)
    gbm.save_model('model_train/lgb_model.txt')  # 用于存储训练出的模型
    preds = gbm.predict(test_x, num_iteration=gbm.best_iteration)
    y_pred = [list(x).index(max(x)) for x in preds]

    return y_pred


# 加载模型
def model_load(realtime_data):
    load_lgb = lgb.Booster(model_file='model_train/lgb_model.txt')
    print("load model_lgb successfully")
    real_pred = load_lgb.predict(realtime_data, num_iteration=load_lgb.best_iteration)
    real_preds = [list(x).index(max(x)) for x in real_pred]
    print("predict successfully")

    return real_preds


if __name__ == '__main__':
    real_df = pd.read_csv("real_df.csv")
    print(real_df)
    print(model_load(real_df))
