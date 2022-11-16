#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Describe :
@Author : James Jun
@Date : 
'''
from main_smd import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type=int, default=32)
    parser.add_argument('-epoch', help='train epoch', type=int, default=1)
    parser.add_argument('-lr', help='learing rate', type=int, default=1e-3)

    parser.add_argument('-slide_win', help='slide_win', type=int, default=30)
    parser.add_argument('-slide_stride', help='slide_stride', type=int, default=1)
    parser.add_argument('-dataset', help='wadi / swat', type=str, default='WT/WT23')
    parser.add_argument("-group", type=str, default="1-1", help="Required for SMD dataset. <group_index>-<index>")
    parser.add_argument('-normalize', type=str2bool, default=True)
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.1)
    parser.add_argument('-smooth', help='Kalman', type=str, default='Kalman')

    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type=int, default=12)

    parser.add_argument('-load_model_path', help='output id', type=str, default='02092021_233355')
    parser.add_argument('-scale_scores', type=str2bool, default=True) # 是否对score进行标准化

    # -- Model params ---
    # 1D conv layer
    parser.add_argument("-kernel_size", type=int, default=7)
    # STGAT layers
    parser.add_argument("-layer_numb", type=int, default=2)

    # LSTM layer
    parser.add_argument("-lstm_n_layers", type=int, default=2)
    parser.add_argument("-lstm_hid_dim", type=int, default=64)
    # Forecasting Model
    parser.add_argument("-fc_n_layers", type=int, default=3)
    parser.add_argument("-fc_hid_dim", type=int, default=150)
    # Reconstruction Model
    parser.add_argument("-recon_n_layers", type=int, default=2)
    parser.add_argument("-recon_hid_dim", type=int, default=150)
    # Other
    parser.add_argument("-alpha", type=float, default=0.2)
    parser.add_argument("-dropout", type=float, default=0.3)

    # 评估算法参数
    parser.add_argument("-dynamic_pot", type=str2bool, default=False)
    parser.add_argument("-level", type=float, default=None)
    parser.add_argument("-q", type=float, default=None)
    parser.add_argument("-gamma", type=float, default=1)
    parser.add_argument("-with_adjust", type=str2bool, default=False)

    parser.add_argument('-batch_train_id', help='train id', type=str, default='03112022_2132')
    parser.add_argument('-batch_train_id_back', help='train id', type=str, default='init')

    args = parser.parse_args()

    random.seed(args.random_seed)               # 设置随机数生成器的种子，是每次随机数相同
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)         # 为CPU设置种子用于生成随机数,以使得结果是确定的
    torch.cuda.manual_seed(args.random_seed)    # 为GPU设置种子用于生成随机数,以使得结果是确定的
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False      # 网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速，适用场景是网络结构以及数据结构固定
    torch.backends.cudnn.deterministic = True   # 为True的话，每次返回的卷积算法将是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的
    os.environ['PYTHONHASHSEED'] = str(args.random_seed) # 为0则禁止hash随机化，使得实验可复现。

    dataset = "WT"
    feature = ["env_temp", "hub_speed", "wind_speed", "f_temp", "b_temp", "power", "grid_power", "fb_diff", "fe_diff","be_diff"]

    args.dataset = "WT"
    args.variable = "WT23"
    args.group = args.variable
    best_path = "20221112_wt"
    args.batch_train_id = best_path
    feature_number = 10
    args.select = []

    for currentId in range(666, 1024):
        current_id_bin = "{:0>10b}".format(currentId)
        # print(current_id_bin)
        args.batch_train_id = best_path + "_" +current_id_bin
        temp = []
        for i in range(10):
            if current_id_bin[i] == '1':
                temp.append(i)
        args.select = temp
        main = Main(args)
        main.run()


    # args.select = list(range(feature_number))
    # main = Main(args)
    # main.run()


    # for j in range(feature_number):
    #     best_f1 = 0
    #     best_id = -1
    #     for i in range(feature_number):
    #         if i in temp:
    #             continue
    #         args.batch_train_id = f"{best_path}_{i}"
    #         args.select = [i]+temp
    #         main = Main(args)
    #         main.run()
    #         f1 = get_f1(f"output/{args.dataset}/{args.variable}/{args.batch_train_id}/summary_file.txt")
    #         if best_f1 < f1:
    #             best_f1 = f1
    #             best_id = i
    #     best_path = best_path+f"_{best_id}"
    #     temp = temp + [best_id]