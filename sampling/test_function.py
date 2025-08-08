from glob import glob
import pandas as pd
from scipy.stats import f_oneway

def test_function(data_path):
    data_freenet = []
    data_reversed = []
    data_random = []
    for file in glob(data_path):
    #   print(file)
        data_freenet.append(pd.read_csv(file, delimiter=',', ).FreeNet_delta_ranks)
        data_reversed.append(pd.read_csv(file, delimiter=',', ).Reversed_delta_ranks)
        data_random.append(pd.read_csv(file, delimiter=',', ).Random_delta_ranks)
    v1 = f_oneway(*data_freenet)
    v2 = f_oneway(*data_reversed)
    v3 = f_oneway(*data_random)
    print(f"F-statistic: {v1.statistic:.4f}, p-value: {v1.pvalue:.4e}")
    print(f"F-statistic: {v2.statistic:.4f}, p-value: {v2.pvalue:.4e}")
    print(f"F-statistic: {v3.statistic:.4f}, p-value: {v3.pvalue:.4e}")