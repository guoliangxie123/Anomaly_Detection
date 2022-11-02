# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : additonal_method.py
# Time       ：2022/10/24 14:35
# Author     ：author name
# version    ：python 3.6
# Description：
"""

import numpy as np
import time
# import pandas as pd
# data1 = pd.read_csv('data_20221023.csv')
# data2 = pd.read_csv('data_20221024.csv')
# data = pd.concat([data1,data2])

def func_timer(function):
    '''
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    '''
    def function_timer(*args, **kwargs):
        print ('[Function: {name} start...]'.format(name = function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ('[Function: {name} finished, spent time: {time:.2f}s]'.format(name = function.__name__,time = t1 - t0))
        return result
    return function_timer

def get_distance(data, model):
    distance = []
    for i in range(len(data)):
        k = np.array(data.iloc[i])
        j = model.cluster_centers_[model.labels_[i]]
        distance.append(np.linalg.norm(k - j))

    return distance


#
# def center_distance(new_data, model):
#     dit = []
#     k = np.array(new_data)
#     for i in model.cluster_centers_:
#         dit.append(np.linalg.norm(k - i))


@func_timer
def get_startpoint(data):
    res = []
    startpoint = -1
    for i in range(len(data)):
        try:
            if startpoint != -1 and i - startpoint <= 380:
                continue

            if 1e-5 < data['预压升降位置'][i] < 1e-2 \
                    and data['预压升降位置'][i] < data['预压升降位置'][i + 1] \
                    and -4 < data['预压升降扭矩'][i] < 0 \
                    and data['预压升降速度'][i] > 1e-2:
                startpoint = i
                res.append([data['时间'][i],data['预压升降位置'][i:i + 380].values
                            ,data['预压升降速度'][i:i + 380].values
                            ,data['预压升降扭矩'][i:i + 380].values])
        except:
            continue
    return res



if __name__ == '__main__':
    print([1, 2, 3] + [2, 3, 4])