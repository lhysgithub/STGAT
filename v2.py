import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np

def get_f1(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["bf_result"]["f1"]
    return f1

if __name__ == '__main__':
    single_f1 = []
    double_f1 = []
    for id in range(10):
        currentId = pow(2, id)
        current_id_bin = "{:0>10b}".format(currentId)
        file_name = f"./output/WT/WT23/{current_id_bin}/summary_file.txt"
        single_f1.append(get_f1(file_name))

        if current_id_bin[7] == '0':
            current_id_bin_l = list(current_id_bin)
            current_id_bin_l[7] = '1'
            current_id_bin = "".join(current_id_bin_l)
        file_name = f"./output/WT/WT23/{current_id_bin}/summary_file.txt"
        double_f1.append(get_f1(file_name))

    file_name = f"./output/WT/WT23/1111111111/summary_file.txt"
    all_f1 = [get_f1(file_name)]

    plt.figure()
    plt.scatter(np.arange(len(single_f1)),single_f1,c="b",label='single',marker="^")
    plt.scatter(np.arange(len(double_f1)), double_f1, c="r", label='double')
    plt.scatter([2], all_f1, c="g", label='all')
    plt.legend(loc='best', fontsize=8)
    plt.savefig(f"./f1_tendency_bf_result.pdf")