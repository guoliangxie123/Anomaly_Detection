# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : test_STL.py
# Time       ：2022/10/10 8:39
# Author     ：author name
# version    ：python 3.6
# Description：
"""

import matplotlib

matplotlib.use('TkAgg')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
import random
from sklearn.ensemble import IsolationForest
import holoviews as hv

hv.extension('bokeh')


def generate_data():
    data = pd.DataFrame([10, 10, 10, 10, 10, 10, 15, 20, 20, 20, 40, 30, 20, 10] * 100, columns=['Torque'])
    data['Torque'] = data.Torque.apply(lambda x: x + random.uniform(-1, 1))
    tor_li = data.Torque.values
    for i in range(len(tor_li)):
        if 20 < i < 70:
            tor_li[i] += random.uniform(-5, 5)
        if 400 < i < 470:
            tor_li[i] += 10
        if 650 < i < 770:
            tor_li[i] -= 10
        if 800 < i < 900:
            tor_li[i] = 0
        if i == 1200:
            tor_li[i] += 60
        if i == 1000:
            tor_li[i] += 30
        if i == 512:
            tor_li[i] -= 30
    return data


def signal_stl(data, period):
    stl = STL(data, period, robust=True)
    res_robust = stl.fit()
    seasonal, trend, resid = res_robust.seasonal, res_robust.trend, res_robust.resid
    res_robust.plot().show()
    return seasonal, trend, resid, res_robust


# 模式的异常检测
def outlier_detection1(data, n=3):
    outlier_x = []
    outlier_y = []
    data_mean = np.mean(data)
    data_std = np.std(data)
    threshold1 = data_mean + n * data_std
    threshold2 = data_mean - n * data_std
    # print(threshold1,threshold2)
    # print(max(trend_data),min(trend_data))
    for i in range(len(data)):
        if (data[i] < threshold2) | (data[i] > threshold1):
            outlier_x.append(i)
            outlier_y.append(data[i])
    return outlier_x, outlier_y


# 点的异常检测
def outlier_detection2(data):
    outlier_x = []
    outlier_y = []
    iforest_model = IsolationForest(n_estimators=300, contamination=0.1, max_samples=700)
    iforest_ret = iforest_model.fit_predict(data.values.reshape(-1, 1))
    for i in range(len(data)):
        if iforest_ret[i] == -1:
            outlier_x.append(i)
            outlier_y.append(data[i])
    return outlier_x, outlier_y


if __name__ == '__main__':
    data = generate_data()
    trend_data = signal_stl(data, 14)[1]
    # print(trend_data)
    resid_data = signal_stl(data, 14)[2]
    # print(resid_data)
    # # print(trend_data)
    # plt.plot(range(len(trend_data)),trend_data)
    outlier_x1, outlier_y1 = outlier_detection1(trend_data, n=1)
    outlier_x2, outlier_y2 = outlier_detection2(resid_data)
    # print(outlier_x)
    anomalies1 = [[idx, value] for idx, value in zip(outlier_x1, outlier_y1)]
    anomalies2 = [[idx, value] for idx, value in zip(outlier_x2, outlier_y2)]
    print(anomalies2)
    # plt.figure(figsize=(20,4))
    # plt.plot(range(len(resid_data)),resid_data)
    # plt.scatter(outlier_x2,outlier_y2,c='r')

    plt.show()

# print(tor_li)

# iforest_model = IsolationForest(n_estimators=300,contamination=0.1,max_samples=700)
# iforest_ret = iforest_model.fit_predict(tor_li.reshape(-1,1))
# print(iforest_ret)
# iforest_df = pd.DataFrame()
# iforest_df['value'] = data['Torque']
# iforest_df['anomaly'] = [1 if i==-1 else 0 for i in iforest_ret]
# print(iforest_df.head())


# anomalies = [[ind, value] for ind, value in zip(iforest_df[iforest_df['anomaly']==1].index, iforest_df.loc[iforest_df['anomaly']==1,'value'])]
# fig = (hv.Curve(iforest_df['value'],label="Torque").opts(color='blue') * hv.Points(anomalies, label="Detected Points").opts(color='red', legend_position='bottom', size=2, title="Isolation Forest - Detected Points"))\
#     .opts(opts.Curve(xlabel="Time", ylabel="Torque", width=700,height=400,tools=['hover'],show_grid=True))
# fig.show_stats()

#
#
