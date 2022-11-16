import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib.ticker import MultipleLocator

metric = "f1"
# metric = "threshold"
# aggregate = "mean"
aggregate = "max"

# dataset = "SMD"
# entity = "1-2"
# compose_number = 38
# feature_number = 38
# base_path = "20221108_smd"

dataset = "SMAP"
entity = "A-5"
compose_number = 25
feature_number = 25
base_path = "20221113_smap"

# dataset = "WT"
# entity = "WT23"
# compose_number = 10
# feature_number = 10
# base_path = "20221112_wt"


class Composition:
   def __init__(self,com,f1,t):
       self.com = com
       self.f1 = f1
       self.t = t


def get_f1(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["bf_result"]["f1"]
    return f1


def get_threshold(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["max_val/train_result"]["threshold"]
    return f1


def get_loss(file_name):
    return np.load(file_name)


def get_data(metric):
    f1s = []
    best_path = base_path
    for k in range(feature_number):
        file_name = f"./output/{dataset}/{entity}/{best_path}_{k}/summary_file.txt"
        f1 = get_f1(file_name)
        ree = get_threshold(file_name)
        loss = get_loss(f"./output/{dataset}/{entity}/{best_path}_{k}/val_mean_loss.npy")
        if metric=="f1":
            f1s.append(f1)
        elif metric == "ree":
            f1s.append(ree)
        elif metric == "loss":
            f1s.append(loss)
    return f1s

if __name__ == '__main__':
    f1s = get_data("f1")
    ree = get_data("ree")
    loss = get_data("loss")

    plt.figure()
    plt.plot(np.arange(len(f1s))+1, f1s, c="b", label=f'f1', marker="^")
    # plt.plot(np.arange(len(ree)) + 1, ree, c="r", label=f'ree', marker="o")
    plt.plot(np.arange(len(loss)) + 1, loss, c="g", label=f'loss', marker="<")
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.grid(linestyle="-.")
    plt.legend(loc='best', fontsize=8)
    plt.title(f"{dataset}_{entity}_{aggregate}_test_aux")
    plt.savefig(f"analysis/{dataset}_{entity}_{compose_number}_{aggregate}_test_aux.pdf")
