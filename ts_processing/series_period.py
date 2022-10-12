import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.fftpack import fft, fftfreq
import random


class PeriodDetection(object):

    def __init__(self, data, top_k_season):
        self.data = data
        self.top_k_season = top_k_season

    # 傅里叶变化，求出TOP3的最大振幅以及对应的周期
    def top3_amplitude(self):
        fft_series = fft(self.data)
        power = np.abs(fft_series)
        sample_freq = fftfreq(fft_series.size)

        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        powers = power[pos_mask]

        top_k_idxs = np.argpartition(powers, -self.top_k_season)[-self.top_k_season:]
        top_k_power = powers[top_k_idxs]
        fft_periods = (1 / freqs[top_k_idxs]).astype(int)

        return top_k_power, fft_periods

    # Expected time period
    def lag_periods(self):
        max_cor_period = []
        fft_periods = self.top3_amplitude()[1]
        for lag in fft_periods:
            acf_score = acf(self.data, nlags=lag)[-1]
            if (len(max_cor_period) == 0) or (acf_score > max_cor_period[-1][1]):
                max_cor_period.append([lag, acf_score])

        return max_cor_period[-1]


def is_period_series(max_cor_period):
    if max_cor_period > 0.8:
        return True
    else:
        return False


if __name__ == '__main__':
    data = pd.DataFrame([10, 20, 30, 40] * 100, columns=['Torque'])
    data['Torque'] = data.Torque.apply(lambda x: x + random.uniform(-1, 1))
    p1 = PeriodDetection(data.Torque.values, 3)
    periods = p1.lag_periods()
    result = is_period_series(periods[1])

    print(periods)
    print(result)
