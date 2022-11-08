import pandas as pd
import random

if __name__ == '__main__':

    for id in range(10):
        currentId = pow(2, id)
        current_id_bin = "{:0>10b}".format(currentId)
        print(current_id_bin)

    dataset = "WT/WT23"
    feature = ["env_temp","hub_speed","wind_speed","f_temp","b_temp","power","grid_power","fb_diff","fe_diff","fb_diff"]
    feature2 = random.sample(feature,4)
    train_orig = pd.read_csv(f'./data/{dataset}/train_orig_23.csv', sep=',').dropna(axis=0)
    # train_orig.drop(labels=["env_temp","hub_speed","wind_speed","f_temp","b_temp"],axis=1,inplace=True)
    train_orig.drop(labels=feature2, axis=1, inplace=True)
    train_orig.to_csv(f'./data/{dataset}/train_orig.csv',index=None)
    test_orig = pd.read_csv(f'./data/{dataset}/test_orig_23.csv', sep=',').dropna(axis=0)
    # test_orig.drop(labels=["env_temp","hub_speed","wind_speed","f_temp","b_temp"], axis=1, inplace=True)
    test_orig.drop(labels=feature2, axis=1, inplace=True)
    test_orig.to_csv(f'./data/{dataset}/test_orig.csv', index=None)