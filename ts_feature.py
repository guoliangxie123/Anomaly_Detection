# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : ts_feature.py
# Time       ：2022/10/9 17:11
# Author     ：author name
# version    ：python 3.6
# Description：
"""

from operator import ge
import pandas as pd
import math


# 信号时域特征的提取

def get_time_domain_features(data):
    """

    Args:
        data: 信号数据

    Returns:

    """

    x_rms = 0
    absXbar = 0
    x_r = 0
    S = 0
    K = 0
    k = 0
    x_rms = 0
    fea = []
    len_ = len(data.iloc[0, :])
    mean_ = data.mean(axis=1)  # 1.均值
    var_ = data.var(axis=1)  # 2.方差
    std_ = data.std(axis=1)  # 3.标准差
    max_ = data.max(axis=1)  # 4.最大值
    min_ = data.min(axis=1)  # 5.最小值
    x_p = max(abs(max_[0]), abs(min_[0]))  # 6.峰值
    for i in range(len_):
        x_rms += data.iloc[0, i] ** 2
        absXbar += abs(data.iloc[0, i])
        x_r += math.sqrt(abs(data.iloc[0, i]))
        S += (data.iloc[0, i] - mean_[0]) ** 3
        K += (data.iloc[0, i] - mean_[0]) ** 4
    x_rms = math.sqrt(x_rms / len_)  # 7.均方根值
    absXbar = absXbar / len_  # 8.绝对平均值
    x_r = (x_r / len_) ** 2  # 9.方根幅值
    W = x_rms / mean_[0]  # 10.波形指标
    C = x_p / x_rms  # 11.峰值指标
    I = x_p / mean_[0]  # 12.脉冲指标
    L = x_p / x_r  # 13.裕度指标
    S = S / ((len_ - 1) * std_[0] ** 3)  # 14.偏斜度
    K = K / ((len_ - 1) * std_[0] ** 4)  # 15.峭度

    fea = [mean_[0], absXbar, var_[0], std_[0], x_r, x_rms, x_p, max_[0], min_[0], W, C, I, L, S, K]
    return fea
