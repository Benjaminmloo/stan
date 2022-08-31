from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import logging
import numpy as np
import pandas as pd
import csv
import time
import os
import random

from stannetflow.synthesizers.nets import SingleTaskNet
from torch.utils.data.dataset import Dataset
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class STANSynthesizer(object):
    def __init__(self, dim_in, dim_window, discrete_columns=[], categorical_columns={},
                 execute_order = None,
                 learning_mode='B', arch_mode='A',
                 data_path='./'):
        """

        @param dim_in: number of columns the input data has
        @param dim_window: number of entries per window
        @param discrete_columns: list of columns belonging to one hot groups
        @param categorical_columns: list of 'categorical columns
        @param execute_order: the order the model will train each column in
        @param learning_mode: The 'mode', 'A' includes current row 'B' mode masks out the values on the current row from the window
        @param arch_mode: sets the 'mode' for the encoder architecture selecting 'A' will define it is a none, any thing else sets the architecture defined in the paper
        @param data_path: the path to a 'checkpoints' folder where the model will attempt to save and load checkpoints from
        """
        assert learning_mode in ['A', 'B'], "Unknown Mask Type"
        self.dim_in = dim_in
        self.dim_window = dim_window
        self.execute_order = execute_order
        self.cur_epoch = 0
        self.data_path = data_path

        #######################################################################
        # prepare for discrete columns
        #######################################################################
        self.discrete_columns = discrete_columns
        self.categorical_columns_dim = categorical_columns
        self.discrete_belong = {}

        # self.discrete_dim = {}
        for dis_col in discrete_columns:
            for sub_dis_col in dis_col:
                self.discrete_belong[sub_dis_col] = dis_col[0]
                if dis_col[0] in self.categorical_columns_dim.keys():
                    self.categorical_columns_dim[dis_col[0]] += 1
                else:
                    self.categorical_columns_dim[dis_col[0]] = 1

        self.cont_agents = []
        self.disc_agents = []
        for col_i in range(self.dim_in):
            if col_i in self.discrete_belong.keys():
                if self.discrete_belong[col_i] not in self.disc_agents:
                    self.disc_agents.append(self.discrete_belong[col_i])
            elif col_i in self.categorical_columns_dim.keys():
                self.disc_agents.append(col_i)
            else:
                self.cont_agents.append(col_i)

        #######################################################################
        # initialize models
        ####################################################################### 
        # print('initing', self.dim_window, dim_in)
        if arch_mode == 'A':
            encoder_arch = None
            gmm_arch = ['gmm', 2]
            gmm_lr = 0.02
            dec_arch = ['softmax', 100]
            dec_lr = 0.02
        else:
            encoder_arch = [64, 64, 'M', 128, 128, 'M']
            gmm_arch = ['gmm', 10]
            gmm_lr = 0.001
            dec_arch = ['softmax', 100] 
            dec_lr = 0.01

        self.models = {}
        self.optimizers = {}
        self.schedulers = {}

        if self.execute_order is None:
            self.execute_order = self.cont_agents+self.disc_agents
        # print(execute_order)
        # input()
        p_mask = [0] * dim_in
        for col_i in self.execute_order:
            if col_i in self.cont_agents:
                mask_mode = None if learning_mode == 'A' else p_mask 
                model_i = SingleTaskNet(dim_in=dim_in, dim_out=1,
                                dim_window=dim_window, mask_mode=mask_mode,
                                encoder_arch=encoder_arch,
                                decoder_arch=gmm_arch, model_tag=col_i)
                
                pytorch_total_params = sum(p.numel() for p in model_i.parameters())
                # print(pytorch_total_params)
                optim_i = optim.Adam(model_i.parameters(), lr=gmm_lr) # 0.005
                sched_i = optim.lr_scheduler.StepLR(optim_i, step_size=10, gamma=0.9)

                self.models[col_i] = model_i
                self.optimizers[col_i] = optim_i
                self.schedulers[col_i] = sched_i
            elif col_i in self.disc_agents:
                mask_mode = None if learning_mode == 'A' else p_mask 
                model_i = SingleTaskNet(dim_in=dim_in, dim_out=self.categorical_columns_dim[col_i],
                            dim_window=dim_window, mask_mode=mask_mode,
                            encoder_arch=encoder_arch, 
                            decoder_arch=dec_arch, model_tag=col_i)
                pytorch_total_params = sum(p.numel() for p in model_i.parameters())
                # print(pytorch_total_params)
                optim_i = optim.Adam(model_i.parameters(), lr=dec_lr)
                sched_i = optim.lr_scheduler.StepLR(optim_i, step_size=10, gamma=0.9)

                self.models[col_i] = model_i
                self.optimizers[col_i] = optim_i
                self.schedulers[col_i] = sched_i
            else:
                input('init cols error')
            # add to mask
            if col_i in self.discrete_belong.keys():
                l = self.categorical_columns_dim[col_i]
                # print('l_len', l)
                for j in range(col_i, col_i+l):
                    p_mask[j] = 1
            else:
                p_mask[col_i] = 1

        # for col_i in self.execute_order:
        #     print(col_i, self.models[col_i].mask)
        # input()

        if device.type == 'cuda':
            for col_i in sorted(self.models.keys()):
                self.models[col_i].to(device)
        # and torch.cuda.device_count() > 1:
        #         self.models[col_i] = nn.DataParallel(self.models[col_i])
        


    def _get_variable_i(self, y, col_i):
        if col_i in self.discrete_belong.keys():
            l = self.categorical_columns_dim[col_i]
            # print('l_len', l)
            return y[:, col_i:col_i+l]
        else:
            return y[:, col_i].view(-1, 1)
    
    def _fill_variable_i(self, y, col_i, fill):
        if col_i in self.discrete_belong.keys():
            l = self.categorical_columns_dim[col_i]
            # print('l_len', l)
            y[-1, col_i+fill] = 1 
        elif col_i in self.categorical_columns_dim.keys():
            y[-1, col_i] = fill/self.categorical_columns_dim[col_i]
        else:
            y[-1, col_i] = fill
        return y
    
    def validate_loss(self, train_loader, loaded_ep=0, new_file=False):
        self.loss_file = 'validation_loss.csv' 
        if new_file:
            with open(self.loss_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerows([['epoch','time']+sorted(self.models.keys())]) 
        
        start_time = time.time()
        for step, (batch_X, batch_y) in enumerate(train_loader):
            minibatch = batch_X.view(-1, self.dim_window+1, self.dim_in).to(device)
            for col_i in self.execute_order:
                # print('training', col_i)
                model = self.models[col_i]
                y_ = self._get_variable_i(batch_y, col_i).to(device)
                loss = model.loss(minibatch, y_, bin_type=True).mean()
                # print(col_i, loss)
            if step % 1000 == 0:
                print('batch steps:', step)

        end_time = time.time()
        temp = [loaded_ep, end_time-start_time]
        for col_i in sorted(self.execute_order):
            temp.append(self.models[col_i].get_batch_loss())

        print(temp)
        with open(self.loss_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerows([temp]) 

    def batch_fit(self, train_loader, epochs=1000):
        self.loss_file = 'train_loss.csv' 
        with open(self.loss_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows([['epoch','time']+sorted(self.models.keys())])

        while self.cur_epoch < epochs:
            print('starting epoch %d'%self.cur_epoch)
            start_time = time.time()
            for step, (batch_X, batch_y) in enumerate(train_loader):
                minibatch = batch_X.view(-1, self.dim_window+1, self.dim_in).to(device)
                for col_i in self.execute_order:
                    # print('training', col_i)
                    model = self.models[col_i]
                    optimizer = self.optimizers[col_i]
                    y_ = self._get_variable_i(batch_y, col_i).to(device)

                    optimizer.zero_grad()
                    loss = model.loss(minibatch, y_, bin_type=True).mean()
                    loss.backward()
                    # print(col_i, loss)
                    optimizer.step()
                    
            end_time = time.time()
            temp = [self.cur_epoch, end_time-start_time]
            for col_i in sorted(self.execute_order):
                scheduler = self.schedulers[col_i]
                scheduler.step()
                temp.append(self.models[col_i].get_batch_loss())
                if self.cur_epoch % 5 == 0:
                    self._save_model(col_i, self.cur_epoch)

                self.models[col_i].batch_reset()
            # print(temp)
            with open(self.loss_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerows([temp])

            self.cur_epoch += 1

    def fit(self, X, y, epochs=100):
        # X = torch.unsqueeze(X, 0)
        for col_i in range(self.dim_in):
            model = self.models[col_i]
            optimizer = self.optimizers[col_i]
            y_ = y[:, col_i].view(-1, 1)
            for i in range(epochs):
                optimizer.zero_grad()
                loss = model.loss(X, y_).mean()
                loss.backward()
                optimizer.step()
                # if i % 100 == 0:
                #     logger.info("Model: %d\t"%col_i + "Iter: %d\t"%i + "Loss: %.2f"%loss.data)

    def sample(self, sample_num):
        p_samp = torch.zeros(( (self.dim_window+1) ,self.dim_in))

        gen_buff = []
        for i in range(sample_num):
            # print('input x_i', p_samp.size(), p_samp)
            j = 0
            for col_i in self.execute_order:
                model_i = self.models[col_i]
                sample_i, normal_i = model_i.sample(p_samp)            
                fill_position = self.dim_window*self.dim_in + col_i
                
                p_samp[-1, col_i] = sample_i


            gen_buff.append(p_samp[-1, :])
           
            p_samp = torch.cat((p_samp[1:, :], torch.zeros(1, self.dim_in)), 0)
           
        gen_buff = torch.cat(gen_buff, 0).view(-1, self.dim_in)
      
        
        return pd.DataFrame(gen_buff.numpy())
    
    def time_series_sample(self, time_limit):
        tot_time = 0
        row_num = 0
        generated_rows = []
      
        p_samp = torch.zeros((self.dim_window+1),self.dim_in).to(device)

        gen_buff = []
        while tot_time < time_limit:
            outputs = []
            j = 0
            for col_i in self.execute_order:
                model_i = self.models[col_i]
                # print(p_samp.shape)
                sample_i, normal_i = model_i.sample(p_samp)
                tiktok = 0
                while col_i in self.cont_agents and (sample_i < 0 or sample_i > 1):
                    # print('!!!!', tot_time, col_i, sample_i)
                    if tiktok > 10:
                        sample_i = 0
                        break
                    tiktok += 1
                    sample_i, normal_i = model_i.sample(p_samp)
                # print(p_samp[-1, :],'==>', sample_i)

                if col_i == 0:
                    self._fill_variable_i(p_samp, col_i, int(tot_time/3600)/24.0)
                else:
                    self._fill_variable_i(p_samp, col_i, sample_i)
                    if col_i == 1:
                        tot_time += int(sample_i * 1336)

            gen_buff.append(p_samp[-1, :])
            # print('whole row', tot_time, 'at', p_samp[-1, :])
            # input()
        
            p_samp = torch.cat((p_samp[1:, :], torch.zeros(1, self.dim_in).to(device)), 0)
        gen_buff = torch.cat(gen_buff, 0).view(-1, self.dim_in)

        return pd.DataFrame(gen_buff.cpu().numpy())
    
    def _save_model(self, col_i, epoch):
        """
        Saves all stateful aspects of the model including the model itself, optimizer and scheduler

        Parameters
        ----------
        col_i - index for the column being saved
        epoch - current epoch being saved for labeling
        """
        Path(self.data_path + 'checkpoints/model_%d'%col_i).mkdir(parents=True, exist_ok=True)


        checkpoint = self.data_path + 'checkpoints/model_%d'%col_i + '/epoch_%d.pkl'%epoch

        torch.save({'epoch': self.cur_epoch,
                    'net_state_dict': self.models[col_i].state_dict(),
                    'opt_state_dict': self.optimizers[col_i].state_dict() ,
                    'sch_state_dict': self.schedulers[col_i].state_dict()
                    }, checkpoint)
    
    def _cpu_loading(self, model, checkpoint):
        state_dict = torch.load(checkpoint, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    def _load_model(self, col_i, epoch):
        """
        Loads all stateful aspects of the model including the model itself, optimizer and scheduler

        Parameters
        ----------
        col_i - index for the column being saved
        epoch - current epoch being saved for labeling
        """
        chk_path = self.data_path + 'checkpoints/model_%d'%col_i + '/epoch_%d.pkl'%epoch

        #model.load_state_dict(torch.load(checkpoint))
        # print(name, model, epoch)
        if device.type == 'cuda':
            checkpoint = torch.load(chk_path)
            self.cur_epoch = checkpoint['epoch']
            self.models[col_i].load_state_dict(checkpoint['net_state_dict'])
            self.optimizers[col_i].load_state_dict(checkpoint['opt_state_dict'])
            self.schedulers[col_i].load_state_dict(checkpoint['sch_state_dict'])
        else:
            raise Exception('CPU LOADING NOT AVAILABLE')
            #self._cpu_loading(model, checkpoint)


    def load_model(self, epoch):
        """
        Loads all the columns defined to the current model

        Parameters
        ----------
        epoch - epoch to be loaded
        """
        if isinstance(epoch, dict):
            for col_i in self.execute_order:
                # print('loading', col_i, 'with checkpoint', epoch[col_i])
                self._load_model(col_i, epoch[col_i])
        else:
            for col_i in self.execute_order:
                self._load_model(col_i, epoch)


class STANCustomDataset(Dataset):
    def __init__(self, csv_path, height, width, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = pd.read_csv(csv_path)
        self.height = height
        self.width = width

    def __getitem__(self, index):
        single_image_label = np.asarray(self.data.iloc[index]).reshape(self.height+1,self.width).astype(np.float32)[-1]
        img_as_np = np.asarray(self.data.iloc[index]).reshape(self.height+1,self.width).astype(np.float32)[:-1]
        img_as_tensor = torch.from_numpy(img_as_np)
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data.index)


class STANCustomDataLoader(object):
    def __init__(self, csv_path, height, width, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.dataset = STANCustomDataset(csv_path, height, width)
        self.loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=512, shuffle=True, \
          num_workers=4, pin_memory=True)
    
    def get_loader(self):
        return self.loader

class NetflowFormatTransformer(object):
    def __init__(self):
        pass

    def _map_ip_str_to_int_list(self, ip_str, ipspace=None):
        ip_group = ip_str.split('.')
        label_rt = []
        rt = []
        pw = 1
        # print(ip_str)
        for i in list(reversed(range(len(ip_group)))):
            label_rt.append(int(ip_group[i]))
        for i in range(len(label_rt)):
            rt.append(label_rt[i]/ipspace)
        return rt, label_rt

    # rt seems to value for model label_rt seems to be human readible number
    def _port_number_interpreter(self, port_num, portspace=None):
        rt = [port_num/portspace]

        def get_category(x):
            return (x-1024)//100+1024 if x >= 1024 else x
        label_rt = [get_category(port_num)]
        return rt, label_rt
    
    def rev_port(self, emb):
        pred_num = int(emb*1670)
        interv = (pred_num-1024) * 100 + 1024
        decode_port = pred_num if pred_num < 1024 else np.random.randint(interv, interv+100)
        if decode_port > 65535:
            decode_port = 65535
        return decode_port

    def rev_transfer(self, df, this_ip=None):
        byt_max = 20.12915933105231 # df['log_byt'].max()
        pkt_max = 12.83
        teT_max = 23 # df['teT'].max()
        teDelta_max = 1336 # df['teDelta'].max()
        ip_space = 255
        port_space = 65535

        td_max = 99.996

        if this_ip is None:
            this_ip = random.choice(['42.219.153.159', '42.219.153.16', '42.219.153.165', '42.219.153.170', '42.219.153.174', '42.219.153.179', '42.219.153.187', '42.219.153.190', '42.219.153.193', '42.219.153.198', '42.219.153.210', '42.219.153.214', '42.219.153.216', '42.219.153.220', '42.219.153.221', '42.219.153.23', '42.219.153.238', '42.219.153.241', '42.219.153.246', '42.219.153.250', '42.219.153.35', '42.219.153.36', '42.219.153.45', '42.219.153.47', '42.219.153.5', '42.219.153.53', '42.219.153.59', '42.219.153.60', '42.219.153.71', '42.219.153.75', '42.219.153.80', '42.219.153.81', '42.219.153.82', '42.219.153.83', '42.219.153.9', '42.219.154.124', '42.219.154.134', '42.219.154.145', '42.219.154.152', '42.219.154.155', '42.219.154.18', '42.219.154.181', '42.219.154.184', '42.219.154.185', '42.219.154.189', '42.219.154.191', '42.219.155.115', '42.219.155.123', '42.219.155.128', '42.219.155.132', '42.219.155.19', '42.219.155.25', '42.219.155.27', '42.219.155.30', '42.219.155.68', '42.219.155.69', '42.219.155.72', '42.219.155.86', '42.219.155.87', '42.219.155.89', '42.219.155.91', '42.219.156.188', '42.219.156.190', '42.219.156.194', '42.219.156.227', '42.219.156.237', '42.219.156.240', '42.219.157.13', '42.219.157.220', '42.219.157.246', '42.219.157.28', '42.219.158.162', '42.219.158.163', '42.219.158.169', '42.219.158.205', '42.219.158.209', '42.219.158.211', '42.219.158.217', '42.219.158.223', '42.219.158.224'])

        # print(df.head()) 
        df['raw_scale_byt'] = np.exp(df[2]*byt_max)
        df['raw_scale_pkt'] = np.exp(df[3]*pkt_max)
        buffer = []
        for index, row in df.iterrows():
            line = [int(row[0]*24), row[1]*teDelta_max, int(row['raw_scale_byt']), int(row['raw_scale_pkt']), row[4]*td_max]
            # line = [int(row[0]*24), row[1]*td_max, int(row['raw_scale_byt']*b_max), int(row['raw_scale_pkt']*pkt_max), row[4]*td_max]
            if row[11] == 1:
                line.append(self.rev_port(row[5])) # sp
                line.append(self.rev_port(row[6])) # dp
                line.append(this_ip) #sa
                line.append('.'.join([str(int(da_i*256)) for da_i in row[7:7+4]])) #da
            else:
                line.append(self.rev_port(row[6])) # dp
                line.append(self.rev_port(row[5])) # sp
                line.append('.'.join([str(int(da_i*256)) for da_i in row[7:7+4]])) #da
                line.append(this_ip) #sa
            prt = ['TCP', 'UDP', 'Other']
            if row[13] == 1:
                line.append(prt[0])
            elif row[14] == 1:
                line.append(prt[1])
            else:
                line.append(prt[2])
            buffer.append(line)
        out_df = pd.DataFrame(buffer)
        out_df.columns = ['hour', 'time_delta', 'byt', 'pkt', 'time_duration', 'sp', 'dp', 'sa', 'da', 'pr']

        return out_df
   
    def transfer(self, df):
        df['log_byt'] = np.log(df['byt'])
        df['log_pkt'] = np.log(df['pkt'])
        byt_max = 20.12915933105231 # df['log_byt'].max()
        pkt_max = 12.83
        teT_max = 23 # d
        teDelta_max = 1336 # df['teDelta'].max()
        ip_space = 255
        port_space = 65535

        td_max = 99.996

        this_ip = df.iloc[0]['this_ip']

        #prevent adding non subnet relevant traffic
        if this_ip == 0:
            return None

        buffer = []
        for index, row in df.iterrows():
            # each row: teT, delta_t, byt, in/out, tcp/udp/other, sa*4, da*4, sp_sig/sp_sys/sp_other, dp*3 
            line = [row['teT']/teT_max, row['teDelta']/teDelta_max, row['log_byt']/byt_max,\
                    row['log_pkt']/pkt_max, row['td']/td_max]

            label_line = [row['teT']/teT_max, row['teDelta']/teDelta_max, row['log_byt']/byt_max, \
                          row['log_pkt']/pkt_max, row['td']/td_max]


            # line = [row['teT']/teT_max, row['log_byt']/byt_max]
            # [out, in]
            sip_list, label_sip_list = self._map_ip_str_to_int_list(row['sa'], ip_space)
            dip_list, label_dip_list = self._map_ip_str_to_int_list(row['da'], ip_space)

            spo_list, label_spo_list = self._port_number_interpreter(row['sp'], port_space)
            dpo_list, label_dpo_list = self._port_number_interpreter(row['dp'], port_space)

            # if sender is local select out else select in
            #also ensure that local IP is first
            if row['sa'] == this_ip:
                #line += sip_list + dip_list
                line += spo_list 
                line += dpo_list + dip_list 
                line += [1, 0]
                
                label_line += label_spo_list 
                label_line += label_dpo_list + label_dip_list 
                label_line += [1, 0]
            else:
                line += dpo_list
                line += spo_list + sip_list
                line += [0, 1]

                label_line += label_dpo_list
                label_line += label_spo_list + label_sip_list
                label_line += [0, 1]

            line_pr = []
            if row['pr'] == 'TCP':
                line_pr = [1, 0, 0]
            elif row['pr'] == 'UDP':
                line_pr = [0, 1, 0]
            else:
                line_pr = [0, 0, 1]
            line += line_pr
            label_line += line_pr

            buffer.append(line)

        df = pd.DataFrame(buffer)
        # print(df)
        return df

class STANTemporalTransformer(object):
    def __init__(self, output_file, special_data=None):
        self.output_file = output_file
        self.special_data = special_data
        self.X ,self.y = None, None
        f = open(output_file, "w+")

        f.close()
    
    def push_back(self, df, agg=1, transformer=None):
        df = df.dropna()
        if transformer:
            df = transformer.transfer(df)

        if isinstance(df, pd.DataFrame):
            X, y = self.agg(df, agg=agg)
            X.to_csv(self.output_file, mode='a', header=False, index=False)


    def agg(self, df, agg=None):
        if agg:
            self.X, self.y = self._agg_window(df, agg)
        return self.X, self.y

    def _get_category(self, x):
        return (x-1024)//100+1024 if x >= 1024 else x

    def _ugr16_label(self, row_list):
        new_list = row_list.copy()
        new_list[5] = self._get_category(np.rint(new_list[5] * 65536))
        new_list[6] = self._get_category(np.rint(new_list[6] * 65536))
        new_list[7] = min(np.rint(new_list[7]*255), 255)
        new_list[8] = min(np.rint(new_list[8]*255), 255)
        new_list[9] = min(np.rint(new_list[9]*255), 255)
        new_list[10] = min(np.rint(new_list[10]*255), 255)
        return new_list

    def _agg_window(self, df_naive, agg_size):
        col_num = len(df_naive.columns)
        buffer = [[0]*col_num] * agg_size
        X, y = [], []

        list_naive = df_naive.values.tolist()
        for row in list_naive:
            buffer.append(row)
            row_with_window = []
            for r in buffer[-agg_size-1:]:
                row_with_window += r
            ugr16_label_row = self._ugr16_label(row)
           
            X.append(row_with_window+ugr16_label_row)
            y.append(row)

        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        return X, y