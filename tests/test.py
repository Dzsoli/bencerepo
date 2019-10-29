import numpy as np
import pandas as pd

csv_file = "../../full_data/full_data.csv"
sorted_xlsx = "../../full_data/full_data_sorted.xlsx"
sorted_csv = "../../full_data/full_data_sorted.csv"
loc_sorted_csv = "../../full_data/full_data_sorted_loc.csv"
clean_data_csv = "../../full_data/clean_data.csv"

feetToMeters = lambda feet: float(feet) * 0.3048
converters_dict = {'Local_X': feetToMeters,
                   'Local_Y': feetToMeters,
                   'v_length': feetToMeters,
                   'v_Width': feetToMeters,
                   'v_Vel': feetToMeters,
                   'v_Acc': feetToMeters,
                   'Space_Headway': feetToMeters}


x = pd.read_csv(csv_file, delimiter=',', header=0, index_col=0)
x = x.sort_values(by=['Location', 'Vehicle_ID', 'Frame_ID', 'Total_Frames'])
x.to_csv(loc_sorted_csv)
# x.to_excel(sorted_csv)
# y = pd.read_excel(sorted_csv, nrows=100, index_col=0)
y = pd.read_csv(loc_sorted_csv, delimiter=',', header=0, usecols=range(0, 25), converters=converters_dict, index_col=0)
y.to_csv(clean_data_csv)
# y = y.sort_values(by=["Location"])

# y = pd.read_csv(clean_data_csv, delimiter=',', header=0, nrows=10000)
# z = np.array(y)
#
# y = pd.read_csv(clean_data_csv, delimiter=',', header=0, nrows=100, index_col=0)
# y_2 = pd.read_csv(clean_data_csv, delimiter=',', header=0, nrows=100)
# y_3 = pd.read_csv(clean_data_csv, delimiter=',', header=0, nrows=20)
# y_4 = pd.read_csv(clean_data_csv, delimiter=',', header=0, nrows=20, index_col=0)

# print(np.array(y_3))
# print(np.array(y_4))

y.to_excel("../../full_data/full_data_sorted_sample2.xlsx")
# y_2.to_excel("../../full_data/full_data_sorted_sample2_2.xlsx")

# x = pd.read_csv(csv_file, delimiter=',', header=0, nrows=100)

# x=x.sort_values(by=['Vehicle_ID', 'Frame_ID'])
# print(x)
# x.to_csv("../../full_data/sample_sort0.csv")
# np.savetxt("sample.csv", x, delimiter=",")
