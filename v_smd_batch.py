import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np


def get_f1(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["bf_result"]["f1"]
        # f1 = summary["max_val/train_result"]["threshold"]
    return f1


if __name__ == '__main__':
    f1s = []
    dataset = "SMD"
    subdir = "1-1"
    select = ""
    selected = []
    best_path = "20221108_smd"
    compose_number = 38
    old_best_f1 = 0
    best_f1 = 0
    for i in range(compose_number):
        old_best_f1 = best_f1
        best_f1 = 0
        best_id = -1
        temp = []
        for j in range(38):
            if j in selected:
                temp.append(old_best_f1)
                continue
            file_name = f"./output/{dataset}/{subdir}/{best_path}/summary_file.txt"
            f1 = get_f1(file_name)
            temp.append(f1)
            if best_f1 < f1:
                best_f1 = f1
                best_id = j
        f1s.append(temp)
        best_path = best_path + f"_{best_id}"
        selected = selected + [best_id]

    f1s_np = np.array(f1s)
    f1s_np_max = np.max(f1s_np, axis=1)  # mean or max?

    plt.figure()
    plt.scatter(np.arange(len(f1s_np_max)), f1s_np_max, c="b", label='single', marker="^")
    plt.legend(loc='best', fontsize=8)
    plt.savefig(f"analysis/{dataset}_{compose_number}_f_tendency.pdf")

    marks = [".", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D",
             "d", '$f$']
    colors = ["r", "g", "b", "c", "m", "y", "k", "w"]
    lines = ["-", ":", "--", "-."]
    plt.figure()
    for i in range(8):
        plt.scatter(np.arange(len(f1s_np[i])), f1s_np[i], c=colors[i], label='single', marker=marks[i])
    plt.legend(loc='best', fontsize=8)
    plt.savefig(f"analysis/{dataset}_8_f.pdf")
