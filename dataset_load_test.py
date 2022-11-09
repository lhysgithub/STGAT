import numpy as np
import pandas as pd
import json

if __name__ == '__main__':
    train = np.load(f'./data/smap_msl/train/A-1.npy')
    train_df = pd.DataFrame(train)
    train_df["y"] = np.zeros(train_df.shape[0])

    test = np.load(f'./data/smap_msl/test/A-1.npy')
    test_df = pd.DataFrame(test)
    test_df["y"] = np.zeros(test_df.shape[0])
    # Set test anomaly labels manually
    test_df.iloc[4690:4774, -1] = 1

    # Set test anomaly labels from files
    labels = pd.read_csv(f'./data/smap_msl/labeled_anomalies.csv', sep=",", index_col="chan_id")
    label_str = labels.loc["A-1", "anomaly_sequences"]
    label_list = json.loads(label_str)
    for i in label_list:
        test_df.iloc[i[0]:i[1], -1] = 1
