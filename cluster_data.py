# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : cluster_data.py.py
# Time       ：2022/10/9 16:00
# Author     ：alan.xie
# version    ：python 3.6
# Description：
"""
# Import related libraries
import random

from series_period import *
from ts_feature import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np
import seaborn


class data_preprocessing(object):
    def __init__(self, data, signal):
        self.data = data
        self.signal = signal

    # 判断自相关函数是否大于阈值
    def data_split(self,k=3):
        """

        Args:
            k: TOP K amplitude
            signal: 输入的数字信号

        Returns: 如果是周期信号，则输出周期，否则输出-1；

        """

        p = PeriodDetection(self.signal, k)
        periods = p.lag_periods()
        if is_period_series(periods[1]):
            return periods[0]
        else:
            return -1;

    # 切分周期,命名工步
    def period_split(self):
        """

        Args:
            signal: 周期信号，经过data_split处理后的

        Returns: 切分好顺序的signal；

        """
        k = 0
        n = 1
        li = []
        period = self.data_split()
        print(period)
        for i in range(0, len(self.signal)):
            if i < period * n:
                li.append(k)
            else:
                k += 1
                n += 1
                li.append(k)
        self.data['cell_code'] = li
        return self.data

    # 经过上一函数处理后的数据
    def gen_1data(self):
        result = []
        for i in data.cell_code.unique():
            ts_data = pd.DataFrame(self.data[self.data['cell_code'] == i]['Torque'])
            re_i = [i] + get_time_domain_features(ts_data.T)
            result.append(re_i)
        columns1 = ['cell_code', 'mean', 'absXbar', 'var', 'std', 'x_r', 'x_rms', 'x_p',
                    'max', 'min', 'W', 'C', 'I', 'L', 'S', 'K']
        df = pd.DataFrame(data=result, columns=columns1)

        return df


if __name__ == '__main__':
    data = pd.DataFrame([10, 20, 30, 40] * 100, columns=['Torque'])
    data['Torque'] = data.Torque.apply(lambda x: x + random.uniform(-1, 1))
    # print(data)

    d1 = data_preprocessing(data, data.Torque.values)
    print(d1.period_split())
    print(d1.gen_1data())
