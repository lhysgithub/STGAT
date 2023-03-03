import os

dir_path = "data/ServerMachineDataset/train"

for file_name in os.listdir(dir_path):
    if file_name.endswith(".txt"):
        dataset = file_name.split(".txt")[0]
        dataset = dataset.replace("machine-", "")
        # print(dataset)
        # if dataset == "1-1" or dataset == "1-2":
        #     continue
        os.system(f"python main_smd_baseline.py -group {dataset}")