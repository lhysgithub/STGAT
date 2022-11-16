#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@Describe :
@Author : James Jun
@Date : 
'''
from main_smd import *

class Main_test(Main):
    def train(self):
        model = self.stgat

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        forecast_criterion = nn.MSELoss().to(self.config.device)
        recon_criterion = nn.MSELoss().to(self.config.device)

        train_loss_list = []

        for epoch in range(self.config.epoch):
            model.train()
            acu_loss = 0
            time_start = time.time()
            # check fc tc edge 是否有问题？ lhy 没问题
            for x, y, attack_labels, fc_edge_index, tc_edge_index in tqdm(self.train_dataloader):
                x, y, fc_edge_index, tc_edge_index = [item.float().to(self.config.device) for item in [x, y,fc_edge_index, tc_edge_index]]
                x = x.permute(0,2,1) # lhy why?
                y = y.unsqueeze(1) # lhy why?

                # 正向传播
                recons = model(x, fc_edge_index, tc_edge_index)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims] # 将输入数据处理成与重建输出相同的形状,如果为NASA数据集则只取第一个维度的数据
                    y = y[:, :, self.target_dims].squeeze(-1) # 将输入数据处理成与预测输出相同的形状,如果未NASA数据集则只取第一个维度的数据

                if y.ndim == 3:
                    y = y.squeeze(1)
                # recon_loss = torch.sqrt(recon_criterion(x[self.config.main_var_id], recons[self.config.main_var_id]))
                recon_loss = torch.sqrt(recon_criterion(x, recons)) # recon_criterion=nn.MSELoss()  + scipy.stats.entropy(x, recons)
                loss = recon_loss


                # 方向梯度下降
                optimizer.zero_grad()  # 梯度清零
                loss.backward()  # 根据误差函数求导
                optimizer.step()  # 进行一轮梯度下降计算

                acu_loss += loss.item()

            time_end = time.time()

            # validation
            model.eval()
            val_loss = []
            for x, y, attack_labels, fc_edge_index, tc_edge_index  in tqdm(self.val_dataloader):
                x, y, fc_edge_index, tc_edge_index  = [item.float().to(self.config.device) for item in [x, y, fc_edge_index, tc_edge_index ]]
                x = x.permute(0,2,1)
                y = y.unsqueeze(1)

                # 正向传播
                recons = model(x, fc_edge_index, tc_edge_index)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)

                if y.ndim == 3:
                    y = y.squeeze(1)

                # recon_loss = torch.sqrt(recon_criterion(x[self.config.main_var_id], recons[self.config.main_var_id]))
                recon_loss = torch.sqrt(recon_criterion(x, recons))

                loss = recon_loss
                val_loss.append(loss.detach().cpu().numpy())

            train_average_loss = acu_loss / len(self.train_dataloader)
            train_loss_list.append(np.atleast_2d(np.array(train_average_loss)))

            val_mean_loss = np.mean(np.array(val_loss))
            print('# epoch:{}/{} , train loss:{} , val loss :{} , cost:{}'.format
                  (epoch, self.config.epoch, train_average_loss, val_mean_loss, (time_end - time_start)))


            np.save(f"{self.save_path}/val_mean_loss.npy", val_mean_loss)

            self.early_stopping(val_mean_loss, model)
            if self.early_stopping.early_stop:
                print('Early stopping')
                train_loss = np.concatenate(train_loss_list)
                np.save(f"{self.save_path}/train_loss.npy",train_loss)
                break

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

    variables = ["A-1", "A-2", "A-3", "A-4", "A-5", "A-6", "A-7", "A-8", "A-9", "B-1", "C-1", "D-1", "D-2", "D-3",
                 "D-4", "D-5", "D-6", "D-7", "D-8", "D-9", "D-10", "D-11", "D-12", "D-13", "D-14", "D-15", "D-16",
                 "E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7", "E-8", "E-9", "E-10", "E-11", "E-12", "E-13"]
    args.dataset = "SMAP"
    args.variable = "A-5"
    args.group = args.variable
    best_path = "20221113_smap"
    args.batch_train_id = best_path
    args.main_var_id = 0
    feature_number = 25
    for i in range(feature_number):
        if i != args.main_var_id:
            args.batch_train_id = f"{best_path}_{i}"
            args.select = [i] + [args.main_var_id]
            main = Main_test(args)
            main.run()
        else:
            args.batch_train_id = f"{best_path}_{i}"
            args.select = [args.main_var_id]
            main = Main_test(args)
            main.run()

    # args.select = list(range(feature_number))
    # main = Main(args)
    # main.run()

    # args.select = []
    # temp = []
    # for j in range(feature_number):
    #     best_f1 = 0
    #     best_id = -1
    #     for i in range(feature_number):
    #         if i in temp:
    #             continue
    #         args.batch_train_id = f"{best_path}_{i}"
    #         args.select = [i]+temp
    #         main = Main_test(args)
    #         main.run()
    #         f1 = get_f1(f"output/{args.dataset}/{args.variable}/{args.batch_train_id}/summary_file.txt")
    #         if best_f1 < f1:
    #             best_f1 = f1
    #             best_id = i
    #     best_path = best_path+f"_{best_id}"
    #     temp = temp + [best_id]



