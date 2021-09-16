import os
import numpy as np
import pandas as pd

# dataset_info = pd.read_csv("ds_info.csv")
# a = dataset_info["dataset"].values
# dataset_names = np.loadtxt("dn.csv", "str")

# for index, row in dataset_info.iterrows():
#     print(row["dataset"])

# results = pd.DataFrame(index = dataset_info["dataset"], columns = ["val_acc", "time_train", "time_val"], data = 0)

# results.loc["ACSF1", "val_acc"] = 1.1

# results.to_csv("test.csv")

for filename in os.listdir("../results"):
    if filename.endswith(".csv"):
        print(filename)