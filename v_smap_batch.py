import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np

metric = "f1"  # "threshold"

def get_f1(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["bf_result"][metric]
        # f1 = summary["max_val/train_result"][metric]
    return f1


if __name__ == '__main__':
    f1s = []
    dataset = "SMD"
    subdir = "1-1"
    select = "0_28_5_27_30_34_22_37_3_35_18_15_12_24_17_16_33_25_36_21_13_23_7_4_32_9_10_14_1_20_29_11_6_26_2_8_31_19"
    selected = []
    best_path = "20221108_smd"
    compose_number = 8
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
            file_name = f"./output/{dataset}/{subdir}/{best_path}_{j}/summary_file.txt"
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
    plt.scatter(np.arange(len(f1s_np_max)), f1s_np_max, c="b", label='tendency', marker="^")
    plt.legend(loc='best', fontsize=8)
    plt.savefig(f"analysis/{dataset}_{compose_number}_{metric}_tendency.pdf")

    marks = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D",
             "d", ".", '$f$']
    colors = ["r", "g", "b", "c", "m", "y", "k", "w"]
    lines = ["-", ":", "--", "-."]
    plt.figure()
    for i in range(compose_number):
        plt.scatter(np.arange(len(f1s_np[i])), f1s_np[i], c=colors[i], label=f"{i}", marker=marks[i])
    plt.legend(loc='best', fontsize=8)
    plt.savefig(f"analysis/{dataset}_{compose_number}_{metric}_discrete.pdf")
