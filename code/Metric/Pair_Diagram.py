import pandas as pd
import matplotlib.pyplot as plt

def pairwise_diagram(directory, csv1, csv2, col):
    result1 = pd.read_csv("{}/{}".format(directory, csv1))
    result2 = pd.read_csv("{}/{}".format(directory, csv2))
    
    col1 = result1[col]
    col2 = result2[col]
    s = min(col1.min(), col2.min())
    e = max(col1.max(), col2.max())
    
    # Plot the diagram
    # Labels on axes
    plt.xlabel(csv1)
    plt.ylabel(csv2)
    
    # texts in the diagram
    upper_left = csv2 if col=="acc_mean" or col =="acc_best" else csv1
    lower_right = csv1 if col=="acc_mean" or col =="acc_best" else csv2
    plt.text(s, e, "{} better here".format(upper_left), ha="left")
    plt.text(e, s, "{} better here".format(lower_right), ha="right")
    
    # diagonal line
    plt.plot([s, e], [s, e])
    
    # points
    plt.scatter(col1, col2, color="r", marker=".", s =  100)
    # plt.show()

if __name__ == "__main__":
    pairwise_diagram("../../results", "rocket_ridgeCV_10000.csv", "DNN_linear_bs128.csv", "time_val")
