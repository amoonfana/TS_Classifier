import argparse
import numpy as np
import torch
import pandas as pd
import time

from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier

from Net.ROCKET import ROCKET

# == notes =====================================================================

# Reproduce the experiments on the UCR archive.
#
# For use with the txt version of the datasets (timeseriesclassification.com)
# and, for datasets with missing values and/or variable-length time series,
# with missing values interpolated, and variable-length time series padded to
# the same length as the longest time series per the version of the datasets as
# per https://www.cs.ucr.edu/~eamonn/time_series_data_2018/.
#
# Arguments:
# -d --dataset_names : txt file of dataset names
# -i --input_path    : parent directory for datasets
# -o --output_path   : path for results
# -n --num_runs      : number of runs (optional, default 10)
# -k --num_kernels   : number of kernels (optional, default 10,000)
#
# *dataset_names* should be a txt file of dataset names, each on a new line.
#
# If *input_path* is, e.g., ".../Univariate_arff/", then each dataset should be
# located at "{input_path}/{dataset_name}/{dataset_name}_TRAIN.txt", etc.

# == parse arguments ===========================================================

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset_names", default = "ds_info.csv")
parser.add_argument("-i", "--input_path", default = "../../UCRArchive_2018")
parser.add_argument("-o", "--output_path", default = "../results")
parser.add_argument("-n", "--num_runs", type = int, default = 1)
parser.add_argument("-k", "--num_kernels", type = int, default = 1_000)

arguments = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# == run =======================================================================

dataset_info = pd.read_csv(arguments.dataset_names)
dataset_names = dataset_info["dataset"].values

results = pd.DataFrame(index = dataset_names,
                       columns = ["acc_mean",
                                  "acc_std",
                                  "acc_best",
                                  "time_train",
                                  "time_val"],
                       data = 0)
results.index.name = "dataset"

print(f"RUNNING".center(80, "="))

for dataset_name in dataset_names:

    print(f"{dataset_name}".center(80, "-"))

    # -- read data -------------------------------------------------------------

    print(f"Loading data".ljust(80 - 5, "."), end = "", flush = True)

    training_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}/{dataset_name}_TRAIN.tsv", dtype=np.float32)
    Y_training, X_training = training_data[:, 0].astype(np.int32), training_data[:, 1:]
    
    # for i in range(X_training.shape[0]):
    #     X_training[i] = (X_training[i] - np.mean(X_training[i]))/np.std(X_training[i])
    X_training = torch.from_numpy(X_training).unsqueeze(1).to(device)
    if Y_training.min(0) == 1:
        Y_training -= 1
    Y_training = torch.from_numpy(Y_training)

    test_data = np.loadtxt(f"{arguments.input_path}/{dataset_name}/{dataset_name}_TEST.tsv", dtype=np.float32)
    Y_test, X_test = test_data[:, 0].astype(np.int32), test_data[:, 1:]

    X_test = torch.from_numpy(X_test).unsqueeze(1).to(device)
    if Y_test.min(0) == 1:
        Y_test -= 1
    Y_test = torch.from_numpy(Y_test)
    
    print("Done.")

    # -- run -------------------------------------------------------------------

    print(f"Performing runs".ljust(80 - 5, "."), end = "", flush = True)

    _results = np.zeros(arguments.num_runs)
    _timings = np.zeros([4, arguments.num_runs]) # trans. tr., trans. te., training, test

    for i in range(arguments.num_runs):
        input_length = X_training.shape[-1]
        model = ROCKET(1, input_length, arguments.num_kernels).to(device)
        # model = torch.load('../my_models/{}.pkl'.format(dataset_name)).to(device)

        # -- transform training ------------------------------------------------
        
        time_a = time.perf_counter()
        with torch.no_grad():
            X_training_transform = model(X_training).cpu().numpy()
        time_b = time.perf_counter()
        _timings[0, i] = time_b - time_a

        # -- transform test ----------------------------------------------------
        
        time_a = time.perf_counter()
        with torch.no_grad():
            X_test_transform = model(X_test).cpu().numpy()
        time_b = time.perf_counter()
        _timings[1, i] = time_b - time_a

        # -- training ----------------------------------------------------------
        torch.cuda.empty_cache()
        
        time_a = time.perf_counter()
        classifier = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True)
        # classifier = RidgeClassifier(normalize = True)
        # classifier = LogisticRegression(penalty='none', max_iter=1000)
        classifier.fit(X_training_transform, Y_training)
        time_b = time.perf_counter()
        _timings[2, i] = time_b - time_a

        # -- test --------------------------------------------------------------
        
        time_a = time.perf_counter()
        _results[i] = classifier.score(X_test_transform, Y_test)
        time_b = time.perf_counter()
        _timings[3, i] = time_b - time_a

    print("Done.")

    # -- store results ---------------------------------------------------------

    results.loc[dataset_name, "acc_mean"] = _results.mean()
    results.loc[dataset_name, "acc_std"] = _results.std()
    results.loc[dataset_name, "acc_best"] = _results.max()
    results.loc[dataset_name, "time_train"] = _timings.mean(1)[[0, 2]].sum()
    results.loc[dataset_name, "time_val"] = _timings.mean(1)[[1, 3]].sum()

print(f"FINISHED".center(80, "="))

results.to_csv(f"{arguments.output_path}/rocket.csv")
