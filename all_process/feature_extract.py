# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : feature_extract.py
# Time       ：2022/10/24 10:15
# Author     ：author name
# version    ：python 3.6
# Description：
"""

import numpy as np
import scipy.stats
from additonal_method import func_timer


class Fea_Extra():
    def __init__(self, Signal, Fs=10):
        self.signal = Signal
        self.Fs = Fs

    def Time_fea(self, signal_):
        """
        提取时域特征 11 类
        """
        N = len(signal_)
        y = signal_
        t_mean_1 = np.mean(y)  # 1_均值（平均幅值）

        t_std_2 = np.std(y, ddof=1)  # 2_标准差

        t_fgf_3 = (np.mean(np.sqrt(np.abs(y)))) ** 2  # 3_方根幅值

        t_rms_4 = np.sqrt((np.mean(y ** 2)))  # 4_RMS均方根

        t_pp_5 = 0.5 * (np.max(y) - np.min(y))  # 5_峰峰值  (参考周宏锑师姐 博士毕业论文)

        # t_skew_6   = np.sum((t_mean_1)**3)/((N-1)*(t_std_3)**3)
        t_skew_6 = scipy.stats.skew(y)  # 6_偏度 skewness

        # t_kur_7   = np.sum((y-t_mean_1)**4)/((N-1)*(t_std_3)**4)
        t_kur_7 = scipy.stats.kurtosis(y)  # 7_峭度 Kurtosis

        t_cres_8 = np.max(np.abs(y)) / t_rms_4  # 8_峰值因子 Crest Factor

        t_clear_9 = np.max(np.abs(y)) / t_fgf_3  # 9_裕度因子  Clearance Factor

        t_shape_10 = (N * t_rms_4) / (np.sum(np.abs(y)))  # 10_波形因子 Shape fator

        t_imp_11 = (np.max(np.abs(y))) / (np.mean(np.abs(y)))  # 11_脉冲指数 Impulse Fator

        t_fea = np.array([t_mean_1, t_std_2, t_fgf_3, t_rms_4, t_pp_5,
                          t_skew_6, t_kur_7, t_cres_8, t_clear_9, t_shape_10, t_imp_11])

        # print("t_fea:",t_fea.shape,'\n', t_fea)
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

        f_14 = (np.sum((y - f_12) ** 3)) / (K * ((np.sqrt(f_13)) ** 3))

        f_15 = (np.sum((y - f_12) ** 4)) / (K * ((f_13) ** 2))

        f_16 = (np.sum(x * y)) / (np.sum(y))

        f_17 = np.sqrt((np.mean(((x - f_16) ** 2) * (y))))

        f_18 = np.sqrt((np.sum((x ** 2) * y)) / (np.sum(y)))

        f_19 = np.sqrt((np.sum((x ** 4) * y)) / (np.sum((x ** 2) * y)))

        f_20 = (np.sum((x ** 2) * y)) / (np.sqrt((np.sum(y)) * (np.sum((x ** 4) * y))))

        f_21 = f_17 / f_16

        f_22 = (np.sum(((x - f_16) ** 3) * y)) / (K * (f_17 ** 3))

        f_23 = (np.sum(((x - f_16) ** 4) * y)) / (K * (f_17 ** 4))

        # f_24 = (np.sum((np.sqrt(x - f_16))*y))/(K * np.sqrt(f_17))    # f_24的根号下出现负号，无法计算先去掉

        # print("f_16:",f_16)

        # f_fea = np.array([f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22, f_23, f_24])
        f_fea = np.array([f_12, f_13, f_14, f_15, f_16, f_17, f_18, f_19, f_20, f_21, f_22, f_23])

        # print("f_fea:",f_fea.shape,'\n', f_fea)
        return f_fea

    def back_feature(self, singal_):
        L = len(singal_)
        work_L1 = [i for i in singal_ if i > 0]
        work_L2 = [i for i in singal_ if i > 40]
        f_24 = len(work_L1) / L
        f_25 = len(work_L2) / L
        w_fea = np.array([f_24, f_25])
        return w_fea

    def Both_Fea(self):
        """
        :return: 时域、频域特征 array
        """
        t_fea = self.Time_fea(self.signal)
        f_fea = self.Fre_fea(self.signal)
        b_fea = self.back_feature(self.signal)
        # print(b_fea,t_fea)
        fea = np.append(np.array(t_fea), np.array(f_fea))
        fea = np.append(fea, np.array(b_fea))
        # print("fea:", fea.shape, '\n', fea)
        return fea


if __name__ == '__main__':
    data = np.array([-1, 2, 3, 4] * 10)
    f = Fea_Extra(data)
    print(f.Both_Fea())
