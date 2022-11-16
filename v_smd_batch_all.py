import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib.ticker import MultipleLocator
from v_smd_batch import get_data

metric = "f1"
# metric = "threshold"
# aggregate = "mean"
aggregate = "max"

# dataset = "SMD"
# entity = "1-2"
# compose_number = 38
# feature_number = 38
# base_path = "20221108_smd"

# dataset = "SMAP"
# entity = "A-6"
# compose_number = 25
# feature_number = 25
# base_path = "20221110_smap"

dataset = "WT"
entity = "WT23"
compose_number = 10
feature_number = 10
gap = 5
# base_path = "20221112_wt"
base_path = "20221110_wt"

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


def get_data_all(metric,base_path):
    f1s = []
    composes = []
    best_path = base_path
    for k in range(1,11):
        temp = []
        temp_com = []
        for currentId in range(1,1024):
            current_id_bin = "{:0>10b}".format(currentId)
            # print(current_id_bin)
            best_path = base_path + "_" + current_id_bin
            selected = []
            for i in range(10):
                if current_id_bin[i] == '1':
                    selected.append(i)
            file_name = f"./output/{dataset}/{entity}/{best_path}/summary_file.txt"
            f1 = get_f1(file_name)
            if len(selected) == k:
                temp.append(f1)
                temp_str = [str(w) for w in selected]
                temp_com.append("_".join(temp_str))
        f1s.append(temp)
    f1s_agg = []
    for l in f1s:
        f1s_agg.append(np.max(l))
    return f1s_agg, f1s, composes



if __name__ == '__main__':
    f1s_agg, f1s, coms = get_data_all("f1", "20221112_wt")
    f1s_agg_max,f1s_max,coms_2 = get_data("f1")



    plt.figure()
    plt.plot(np.arange(len(f1s_agg))+1, f1s_agg, c="b", label=f'f1_all_composition', marker="v")
    plt.plot(np.arange(len(f1s_agg_max)) + 1, f1s_agg_max, c="r", label=f'f1_max_composition', marker="^")
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.grid(linestyle="-.")
    plt.legend(loc='best', fontsize=8)
    plt.title(f"{dataset}_{entity}_{aggregate}_all_tendency")
    plt.savefig(f"analysis/{dataset}_{entity}_{compose_number}_{aggregate}_all_tendency.pdf")

    center = 0.75
    max_lens = 0
    for i in f1s:
        if max_lens < len(i):
            max_lens = len(i)
    mask_base = [True for i in range(max_lens)]
    f1s_base = [center for i in range(max_lens)]

    f1s_new = []
    masks = []
    for l in f1s:
        temp_mask = mask_base.copy()
        temp_f1s = f1s_base.copy()
        for i in range(len(l)):
            temp_mask[i] = False
            temp_f1s[i] = l[i]
        f1s_new.append(temp_f1s)
        masks.append(temp_mask)

    plt.figure()
    sns.set(font_scale=0.8)
    labels = [i+1 for i in range(10)] # , yticklabels=labels
    p = sns.heatmap(f1s_new, cmap="RdBu_r", yticklabels=labels, cbar_kws={"orientation":"horizontal", "label":"f1 score"},vmin=0,vmax=1,center=center,square=False)
    p.set_xlabel("index")
    p.set_ylabel("composition")
    plt.title(f"{dataset}_{entity}_all_f1_heatmap")
    plt.savefig(f"analysis/{dataset}_{entity}_{compose_number}_all_f1_heatmap.pdf")

    # all_f1s = [0]
    # for i in f1s:
    #     all_f1s = all_f1s + i
    # f1s_np = np.array(all_f1s)
    # f1s_np = f1s_np.reshape((32,32))
    # f1s_pd = pd.DataFrame()
    # for i in range(len(f1s_np)):
    #     f1s_pd.insert(len(f1s_pd.columns), i, f1s_np[i])
    #
    # plt.figure()
    # sns.heatmap(f1s_pd)
    # plt.savefig(f"analysis/{dataset}_{entity}_{compose_number}_all_f1_heatmap.pdf")
    # marks = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D",
    #          "d", ".", '$f$']
    # colors = ["r", "g", "b", "c", "m", "y", "k"]
    # lines = ["-", ":", "--", "-."]
    # plt.figure()
    # for i in range(compose_number):
    #     plt.scatter(np.arange(len(f1s_np[i]))+1, f1s_np[i], c=colors[i%len(colors)], label=f"{i}", marker=marks[i%len(marks)])
    # plt.legend(loc='best', fontsize=8)
    # plt.savefig(f"analysis/{dataset}_{entity}_{compose_number}_{metric}_discrete.pdf")
