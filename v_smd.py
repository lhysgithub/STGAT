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
    single_f1 = []
    double_f1 = []
    dataset = "SMD"
    subdir = "1-1"
    for i in range(38):
        current_id_bin = f"20221108_smd_{i}"
        file_name = f"./output/{dataset}/{subdir}/{current_id_bin}/summary_file.txt"
        single_f1.append(get_f1(file_name))

    for i in range(38):
        current_id_bin = f"20221108_smd_0_{i}"
        if i == 0:
            current_id_bin = f"20221108_smd_{i}"
        file_name = f"./output/{dataset}/{subdir}/{current_id_bin}/summary_file.txt"
        double_f1.append(get_f1(file_name))

    dl = np.array(double_f1)
    print(dl.argmax())

    plt.figure()
    plt.scatter(np.arange(len(single_f1)), single_f1, c="b", label='single', marker="^")
    plt.scatter(np.arange(len(double_f1)), double_f1, c="r", label='double')
    plt.legend(loc='best', fontsize=8)
    plt.savefig(f"analysis/{dataset}_2_f.pdf")