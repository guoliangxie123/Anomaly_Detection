from series_period import *
import  pandas as pd


if __name__ == '__main__':

    cPath = 'C:\\Users\\1212214\\Desktop\\23.csv'

    data = pd.read_csv(cPath, delimiter='\t')
    data = data['Poisition'][100000:].values
    p1 = PeriodDetection(data, 3)
    fft_periods = p1.top3_amplitude()[1]
    periods = p1.lag_periods(fft_periods)
    result = is_period_series(periods[1])

    print(fft_periods)
    print(periods)
    print(result)




