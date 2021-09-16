from Metric.Pair_Diagram import pairwise_diagram
from Metric.CD_Diagram import cd_diagram

res_dir = "../results"

# pairwise_diagram(res_dir, "DNN_linear_epoch200.csv", "DNN.csv", "acc_mean")
cd_diagram(res_dir, "../diagrams/cd-diagram.png", "acc_mean")