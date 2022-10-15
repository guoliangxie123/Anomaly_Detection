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

''' ============== 特征提取的类 =====================
时域特征 ：11类
频域特征 : 13类
总共提取特征 ： 24类

参考文献 英文文献 016_C_(Q1 时域和频域共24种特征参数 )  Fault diagnosis of rotating machinery based on multiple ANFIS combination with GAs

'''

import numpy as np

import scipy.stats
import matplotlib.pyplot as plt

class Fea_Extra():
    def __init__(self, Signal, Fs = 25600):
        self.signal = Signal
        self.Fs = Fs

    def Time_fea(self, signal_):
        """
        提取时域特征 11 类
        """
        N = len(signal_)
        y = signal_
        t_mean_1 = np.mean(y)                                    # 1_均值（平均幅值）

        t_std_2  = np.std(y, ddof=1)                             # 2_标准差

        t_fgf_3  = ((np.mean(np.sqrt(np.abs(y)))))**2           # 3_方根幅值

        t_rms_4  = np.sqrt((np.mean(y**2)))                      # 4_RMS均方根

        t_pp_5   = 0.5*(np.max(y)-np.min(y))                     # 5_峰峰值  (参考周宏锑师姐 博士毕业论文)

        #t_skew_6   = np.sum((t_mean_1)**3)/((N-1)*(t_std_3)**3)
        t_skew_6   = scipy.stats.skew(y)                         # 6_偏度 skewness

        #t_kur_7   = np.sum((y-t_mean_1)**4)/((N-1)*(t_std_3)**4)
        t_kur_7 = scipy.stats.kurtosis(y)                        # 7_峭度 Kurtosis

        t_cres_8  = np.max(np.abs(y))/t_rms_4                    # 8_峰值因子 Crest Factor

        t_clear_9  = np.max(np.abs(y))/t_fgf_3                   # 9_裕度因子  Clearance Factor

        t_shape_10 = (N * t_rms_4)/(np.sum(np.abs(y)))           # 10_波形因子 Shape fator

        t_imp_11  = ( np.max(np.abs(y)))/(np.mean(np.abs(y)))  # 11_脉冲指数 Impulse Fator

        t_fea = np.array([t_mean_1, t_std_2, t_fgf_3, t_rms_4, t_pp_5,
                          t_skew_6,   t_kur_7,  t_cres_8,  t_clear_9, t_shape_10, t_imp_11 ])

        #print("t_fea:",t_fea.shape,'\n', t_fea)
        return t_fea

    def Fre_fea(self, signal_):
        """
        提取频域特征 13类
        :param signal_:
        :return:
        """
        L = len(signal_)
        PL = abs(np.fft.fft(signal_ / L))[: int(L / 2)]
        PL[0] = 0
        f = np.fft.fftfreq(L, 1 / self.Fs)[: int(L / 2)]
        x = f
        y = PL
        K = len(y)
        # print("signal_.shape:",signal_.shape)
        # print("PL.shape:", PL.shape)
        # print("L:", L)
        # print("K:", K)
        # print("x:",x)
        # print("y:",y)

        f_12 = np.mean(y)

        f_13 = np.var(y)

        f_14 = (np.sum((y - f_12)**3))/(K * ((np.sqrt(f_13))**3))

        f_15 = (np.sum((y - f_12)**4))/(K * ((f_13)**2))

        f_16 = (np.sum(x * y))/(np.sum(y))

        f_17 = np.sqrt((np.mean(((x- f_16)**2)*(y))))

        f_18 = np.sqrt((np.sum((x**2)*y))/(np.sum(y)))

        f_19 = np.sqrt((np.sum((x**4)*y))/(np.sum((x**2)*y)))

        f_20 = (np.sum((x**2)*y))/(np.sqrt((np.sum(y))*(np.sum((x**4)*y))))

        f_21 = f_17/f_16

        f_22 = (np.sum(((x - f_16)**3)*y))/(K * (f_17**3))

        f_23 = (np.sum(((x - f_16)**4)*y))/(K * (f_17**4))

        #f_24 = (np.sum((np.sqrt(x - f_16))*y))/(K * np.sqrt(f_17))    # f_24的根号下出现负号，无法计算先去掉


        #print("f_16:",f_16)

        #f_fea = np.array([f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22, f_23, f_24])
        f_fea = np.array([f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22, f_23])

        #print("f_fea:",f_fea.shape,'\n', f_fea)
        return f_fea

    def Both_Fea(self):
        """
        :return: 时域、频域特征 array
        """
        t_fea = self.Time_fea(self.signal)
        f_fea = self.Fre_fea(self.signal)
        fea = np.append(np.array(t_fea), np.array(f_fea))
        #print("fea:", fea.shape, '\n', fea)
        return fea
