import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib.ticker import MultipleLocator

metric = "f1"
# metric = "ree"
# metric = "precision"
# metric = "recall"
# aggregate = "mean"
aggregate = "max"

# dataset = "SMD"
# entity = "1-3"
# compose_number = 38
# feature_number = 38
# gap = 2
# base_path = "20221108_smd"

# dataset = "SMAP"
# entity = "A-7"
# compose_number = 25
# feature_number = 25
# gap = 2
# base_path = "20221110_smap"

dataset = "WT"
entity = "WT23"
compose_number = 10
feature_number = 10
gap = 1
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


def get_precision(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["bf_result"]["precision"]
    return f1


def get_recall(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["bf_result"]["recall"]
    return f1


def get_auc(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["bf_result"]["roc_auc"]
    return f1


def get_threshold(file_name):
    with open(file_name) as f:
        summary = json.load(f)
        f1 = summary["max_val/train_result"]["threshold"]
    return f1


def get_data(metric):
    f1s = []
    composes = []
    selected = []
    best_path = base_path
    for i in range(compose_number):
        temp2_f1 = []
        for j in range(feature_number):
            if j in selected:
                continue
            file_name = f"./output/{dataset}/{entity}/{best_path}_{j}/summary_file.txt"
            f1 = get_f1(file_name)
            threshold = get_threshold(file_name)
            precision = get_precision(file_name)
            recall = get_recall(file_name)
            auc = get_auc(file_name)
            if metric == "f1":
                temp2_f1.append(f1)
            elif metric == "ree":
                temp2_f1.append(threshold)
            elif metric == "precision":
                temp2_f1.append(precision)
            elif metric == "recall":
                temp2_f1.append(recall)
            elif metric == "auc":
                temp2_f1.append(auc)
        mean_f1 = np.array(temp2_f1).mean()

        best_f1 = 0
        best_id = -1
        temp_f1 = []
        temp_com = []
        for j in range(feature_number):
            if j in selected:
                temp_f1.append(mean_f1)
                selected_str = [str(k) for k in selected]
                current_composition = "_".join(selected_str)
                temp_com.append(current_composition)
                continue
            file_name = f"./output/{dataset}/{entity}/{best_path}_{j}/summary_file.txt"
            f1 = get_f1(file_name)
            threshold = get_threshold(file_name)
            precision = get_precision(file_name)
            recall = get_recall(file_name)
            auc = get_auc(file_name)
            selected_str = [str(k) for k in selected]
            if len(selected_str) == 0:
                current_composition = "_".join(selected_str) + f"{j}"
            else:
                current_composition = "_".join(selected_str) + f"_{j}"
            temp_com.append(current_composition)
            if metric == "f1":
                temp_f1.append(f1)
            elif metric == "ree":
                temp_f1.append(threshold)
            elif metric == "precision":
                temp_f1.append(precision)
            elif metric == "recall":
                temp_f1.append(recall)
            elif metric == "auc":
                temp_f1.append(auc)
            if best_f1 < f1:
                best_f1 = f1
                best_id = j
        f1s.append(temp_f1)
        composes.append(temp_com)
        best_path = best_path + f"_{best_id}"
        selected = selected + [best_id]
    f1s_np = np.array(f1s)
    coms_np = composes
    f1s_np_max = np.max(f1s_np, axis=1)  # mean or max?
    f1s_np_mean = np.mean(f1s_np, axis=1)  # mean or max?
    if aggregate == "mean":
        f1s_np_max = f1s_np_mean
    return f1s_np_max,f1s_np,coms_np


def render_heatmap(data_np,composition,type):
    data_pd = pd.DataFrame()
    for i in range(len(data_np)):
        data_pd.insert(len(data_pd.columns), i, data_np[i])

    plt.figure()
    # sns.heatmap(data_pd)
    center = np.mean(data_np)#0.5
    labels = []
    temp_labels = [i + 1 for i in range(feature_number)]
    for i in temp_labels:
        if i % gap == 0:
            labels.append(i)
        else:
            labels.append("")
    p = sns.heatmap(data_pd,  cmap="RdBu_r", xticklabels=labels, yticklabels=labels,
                    cbar_kws={"label": f"{type}"}, vmin=0, vmax=1, center=center,
                    square=False)#yticklabels=ylabes,
    p.set_ylabel("index")
    p.set_xlabel("composition")
    plt.title(f"{dataset}_{entity}_{type}_heatmap")
    plt.savefig(f"analysis/{dataset}_{entity}_{compose_number}_{type}_heatmap.pdf")


def render_csv(data_np, composition, type):
    data_com_pd = pd.DataFrame()
    for i in range(len(data_np)):
        data_com_pd.insert(len(data_com_pd.columns), f"{i}_coms", composition[i])
        data_com_pd.insert(len(data_com_pd.columns), f"{i}_f1", data_np[i])
    data_com_pd.to_csv(f"analysis/{dataset}_{entity}_{compose_number}_{type}_composition.csv", float_format='%.2f')


if __name__ == '__main__':
    f1s_aggregated, f1s_np, coms = get_data("f1")
    ree_aggregated, rees_np, _ = get_data("ree")
    precision_aggregated, precision_np, _ = get_data("precision")
    recall_aggregated, recall_np, _ = get_data("recall")
    auc_aggregated, auc_np, _ = get_data("auc")

    render_heatmap(f1s_np, coms, "f1")
    render_csv(f1s_np, coms, "f1")
    render_heatmap(precision_np, coms, "precision")
    render_heatmap(recall_np, coms, "recall")
    # render_heatmap(rees_np, coms, "ree")
    render_heatmap(auc_np, coms, "auc")


    plt.figure()
    plt.plot(np.arange(len(f1s_aggregated)) + 1, f1s_aggregated, c="r", label=f'f1', marker="^")
    plt.plot(np.arange(len(precision_aggregated)) + 1, precision_aggregated, c="g", label=f'precision', marker="o")
    plt.plot(np.arange(len(recall_aggregated)) + 1, recall_aggregated, c="b", label=f'recall', marker="v")
    # plt.plot(np.arange(len(ree_aggregated)) + 1, ree_aggregated, c="k", label=f'ree', marker="<")
    plt.plot(np.arange(len(auc_aggregated)) + 1, auc_aggregated, c="y", label=f'auc', marker=">")
    # plt.plot(np.arange(len(ree_aggregated)) + 1, ree_aggregated, c="r", label=f'ree_aggregated', marker="o")
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.grid(linestyle="-.")
    plt.legend(loc='best', fontsize=8)
    plt.xlabel("composition")
    plt.title(f"{dataset}_{entity}_{aggregate}_tendency")
    plt.savefig(f"analysis/{dataset}_{entity}_{compose_number}_{aggregate}_tendency.pdf")

    marks = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D",
             "d", ".", '$f$']
    colors = ["r", "g", "b", "c", "m", "y", "k"]
    lines = ["-", ":", "--", "-."]
    # plt.figure()
    # for i in range(compose_number):
    #     plt.scatter(np.arange(len(f1s_np[i]))+1, f1s_np[i], c=colors[i%len(colors)], label=f"{i}", marker=marks[i%len(marks)])
    # plt.legend(loc='best', fontsize=8)
    # plt.savefig(f"analysis/{dataset}_{entity}_{compose_number}_{metric}_discrete.pdf")
