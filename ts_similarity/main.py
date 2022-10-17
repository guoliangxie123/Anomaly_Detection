# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : main.py
# Time       ：2022/10/17 16:00
# Author     ：author name
# version    ：python 3.6
# Description：
"""
import numpy as np
np.random.seed(0)
a = np.random.normal(3, 2.5, size=(2, 1000))

# 使用tslearn计算dtw距离
import time as t
time1 = t.time()
from tslearn.metrics import dtw
dist_ts = dtw(a[0], a[1])
time2 = t.time()
cost_ts = time2-time1

# 使用dtw计算dtw距离
time1 = t.time()
from dtw import dtw
from numpy.linalg import norm
dist_dtw, cost, acc_cost, path = dtw(
    a[0].reshape(-1, 1),
    a[1].reshape(-1, 1),
    dist=lambda x, y: norm(x - y, ord=1)
)
time2 = t.time()
cost_dtw = time2-time1

# 使用fastdtw计算dtw距离
time1 = t.time()
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
dist_fast, path1 = fastdtw(a[0], a[1], dist=euclidean)
time2 = t.time()
cost_fast = time2-time1

# 使用dtaidistance计算dtw距离
time1 = t.time()
from dtaidistance import dtw
dist_dtai = dtw.distance_fast(a[0], a[1])
time2 = t.time()
cost_dtai = time2-time1
