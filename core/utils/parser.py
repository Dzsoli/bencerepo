import csv
import numpy as np
import pandas as pd

from core.common import *


def parser():
    #Todo: dataframe groupby https://stackoverflow.com/questions/40498463/python-splitting-dataframe-into-multiple-dataframes-based-on-column-values-and/40498517
    #https://stackoverflow.com/questions/33742588/pandas-split-dataframe-by-column-value/33742822
    #Todo: ezzel meg lehet cisnálni az egész szeparációt is akár
    csv_file = "../../../full_data/full_data.csv"

    feetToMeters = lambda feet: float(feet) * 0.3048
    converters_dict = {'Local_X': feetToMeters,
                       'Local_Y': feetToMeters,
                       'v_length': feetToMeters,
                       'v_Width': feetToMeters,
                       'v_Vel': feetToMeters,
                       'v_Acc': feetToMeters,
                       'Space_Headway': feetToMeters}

    full_data = pd.read_csv(csv_file, delimiter=',', header=0, index_col=0, converters=converters_dict)
    full_data = full_data.sort_values(by=['Location', 'Vehicle_ID', 'Frame_ID', 'Total_Frames'])

    for _, location in full_data.groupby('Location'):
        path = "../../../full_data/"
        name = location['Location'].values[0] + ".csv"
        location.to_csv(path + name)


if __name__ == '__main__':
    parser()
