import time
import math
import pandas as pd
import torch

from models.Model_base import *
from models import LeNet_FashionMNIST, CNN_Cifar10, CNN_Cifar100, Model_adults, Model_purchase
from utils import init_network, test_class_forget, test_client_forget
from dataset.data_utils import *
from algs.fl_base import Base
import torch.optim as optim
import copy
import logging
import matplotlib.pyplot as plt
from utils import *
import random
from models.Model_base import *

class FUSED(Base):
    def __init__(self, args):
        super(FUSED, self).__init__(args)
        self.args = args
        self.log_dir = f"logs/fused_{self.args.data_name}_{self.args.alpha}"
        self.param_change_dict = {}
        self.param_size = {}

    def train_normal(self, global_model, client_all_loaders, test_loaders):
        checkpoints_ls = []
        result_list = []
        param_list = []
        for name, param in global_model.named_parameters():
            # print(name)
            self.param_change_dict[name] = 0
            self.param_size[name] = 0

        for epoch in range(self.args.global_epoch):
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user*self.args.fraction), replace=False))
            select_client_loaders = [client_all_loaders[idx] for idx in selected_clients]
            client_models = self.global_train_once(epoch, global_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)

            global_model = self.fedavg(client_models)

            if self.args.forget_paradigm == 'sample':
            
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model,
                                                                                              self.args, test_loaders)
                print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero,
                                                                                 avg_test_acc))
                result_list.extend(test_result_ls)
            
            elif self.args.forget_paradigm == 'client':
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, global_model, self.args, test_loaders)
                print('Epoch={}, avg_f_acc={}, avg_r_acc={}'.format(epoch, avg_f_acc, avg_r_acc))
                result_list.extend(test_result_ls)
            
            elif self.args.forget_paradigm == 'class':
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args, test_loaders)
                print('Epoch={}, avg_f_acc={}, avg_r_acc={}'.format(epoch, avg_f_acc, avg_r_acc))
                result_list.extend(test_result_ls)

            if self.args.paradigm == 'fused':
                diff_ls = list(self.param_change_dict.values())
                name = list(self.param_change_dict.keys())
                diff_ls_ = [float(i) for i in diff_ls]
                param_list.append(diff_ls_)
                # diff_ls_.append(list(self.param_size.values()))
        df = pd.DataFrame(param_list, columns=name)
        df.to_csv('./results/param_change_{}_distri_{}.csv'.format(self.args.data_name, self.args.alpha))

        torch.save(global_model.state_dict(), 'save_model/global_model_{}.pth'.format(self.args.data_name))

        if self.args.forget_paradigm == 'sample':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc'])
        elif self.args.forget_paradigm == 'client':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss'])
        elif self.args.forget_paradigm == 'class':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss'])
        if self.args.save_normal_result:
            df.to_csv('./results/Acc_loss_fl_{}_data_{}_distri_{}.csv'.format(self.args.forget_paradigm, self.args.data_name, self.args.alpha))

        return global_model, client_models

    def forget_client_train(self, global_model, client_all_loaders, test_loaders):
        global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(self.args.data_name)))
        avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, 1, global_model, self.args,
                                                                  test_loaders)
        print('FUSED-epoch-{}-client forget, Avg_r_acc: {}, Avg_f_acc: {}'.format('xxxx', avg_r_acc,
                                                                                 avg_f_acc))

        fused_model = Lora(self.args, global_model)
        torch.save(fused_model.state_dict(), 'save_model/global_fusedmodel_{}.pth'.format(self.args.data_name))

        checkpoints_ls = []
        result_list = []
        consume_time = 0

        for epoch in range(self.args.global_epoch):
            fused_model.train()
            selected_clients = [i for i in range(self.args.num_user) if i not in self.args.forget_client_idx]
            select_client_loaders = select_part_sample(self.args, client_all_loaders, selected_clients)


            std_time = time.time()
            client_models = self.global_train_once(epoch, fused_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            avg_model = self.fedavg(client_models)
            consume_time += end_time - std_time
            fused_model.load_state_dict(avg_model.state_dict())

            fused_model.eval()

            avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, fused_model, self.args,
                                                                      test_loaders)

            result_list.extend(test_result_ls)

            print('FUSED-epoch-{}-client forget, Avg_r_acc: {}, Avg_f_acc: {}'.format(epoch, avg_r_acc,
                                                                                    avg_f_acc))


        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss'])
        df['Comsume_time'] = consume_time

        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv('./results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm,
                                                                                              self.args.forget_paradigm,
                                                                                              self.args.data_name,
                                                                                              self.args.alpha,
                                                                                              len(self.args.forget_class_idx)))
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'.format(
                        self.args.forget_paradigm,
                        self.args.forget_paradigm,
                        self.args.data_name,
                        self.args.alpha,
                        len(self.args.forget_class_idx), self.args.cut_sample))

        return fused_model

    def forget_class(self, global_model, client_all_loaders, test_loaders):
        checkpoints_ls = []
        result_list = []
        consume_time = 0

        fused_model = Lora(self.args, global_model)
        for epoch in range(self.args.global_epoch):
            fused_model.train()
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user * self.args.fraction), replace=False))

            select_client_loaders = select_part_sample(self.args, client_all_loaders, selected_clients)
            std_time = time.time()

            client_models = self.global_train_once(epoch, fused_model,  select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            fused_model = self.fedavg(client_models)
            consume_time += end_time-std_time
            avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, fused_model, self.args, test_loaders)
            result_list.extend(test_result_ls)
            print('Epoch={}, Remember Test Acc={}, Forget Test Acc={}'.format(epoch, avg_r_acc, avg_f_acc))

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss'])
        df['Comsume_time'] = consume_time

        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm, self.args.forget_paradigm, self.args.data_name, self.args.alpha, len(self.args.forget_class_idx)))
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'.format(self.args.forget_paradigm,
                                                                                     self.args.forget_paradigm,
                                                                                     self.args.data_name,
                                                                                     self.args.alpha,
                                                                                     len(self.args.forget_class_idx), self.args.cut_sample))

        return fused_model

    def forget_sample(self, global_model, client_all_loaders, test_loaders):
        checkpoints_ls = []
        result_list = []
        consume_time = 0

        fused_model = Lora(self.args, global_model)
        for epoch in range(self.args.global_epoch):
            fused_model.train()
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user * self.args.fraction), replace=False))# 将需要遗忘的客户端排除在外

            self.select_forget_idx = list()
            select_client_loaders = list()
            record = -1
            for idx in selected_clients:
                select_client_loaders.append(client_all_loaders[idx])
                record += 1
                if idx in self.args.forget_client_idx:
                    self.select_forget_idx.append(record)
            std_time = time.time()
            client_models = self.global_train_once(epoch, fused_model,  select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            fused_model = self.fedavg(client_models)
            consume_time += end_time-std_time

            avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, fused_model, self.args, test_loaders)
            result_list.extend(test_result_ls)
            print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero, avg_test_acc))

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc'])
        df['Comsume_time'] = consume_time
        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm, self.args.forget_paradigm, self.args.data_name, self.args.alpha, len(self.args.forget_class_idx)))
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'.format(self.args.forget_paradigm,
                                                                                        self.args.forget_paradigm,
                                                                                        self.args.data_name,
                                                                                        self.args.alpha,
                                                                                        len(self.args.forget_class_idx), self.args.cut_sample))

        return fused_model

    def relearn_unlearning_knowledge(self, unlearning_model, client_all_loaders, test_loaders):
        checkpoints_ls = []
        all_global_models = list()
        all_client_models = list()
        global_model = unlearning_model
        result_list = []

        all_global_models.append(global_model)
        std_time = time.time()
        for epoch in range(self.args.global_epoch):
            if self.args.forget_paradigm == 'client':
                select_client_loaders = list()
                for idx in self.args.forget_client_idx:
                    select_client_loaders.append(client_all_loaders[idx])
            elif self.args.forget_paradigm == 'class':
                select_client_loaders = list()
                client_loaders = select_forget_class(self.args, copy.deepcopy(client_all_loaders))
                for v in client_loaders:
                    if v is not None:
                        select_client_loaders.append(v)
            elif self.args.forget_paradigm == 'sample':
                select_client_loaders = list()
                client_loaders = select_forget_sample(self.args, copy.deepcopy(client_all_loaders))
                for v in client_loaders:
                    if v is not None:
                        select_client_loaders.append(v)
            client_models = self.global_train_once(epoch, global_model, select_client_loaders, test_loaders, self.args,
                                                   checkpoints_ls)

            all_client_models += client_models
            global_model = self.fedavg(client_models)
            all_global_models.append(copy.deepcopy(global_model).to('cpu'))
            end_time = time.time()

            consume_time = end_time - std_time

            if self.args.forget_paradigm == 'client':
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, global_model, self.args,
                                                                          test_loaders)
                for item in test_result_ls:
                    item.append(consume_time)
                result_list.extend(test_result_ls)
                df = pd.DataFrame(result_list,
                                  columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss',
                                           'Comsume_time'])
            elif self.args.forget_paradigm == 'class':
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args,
                                                                         test_loaders)
                for item in test_result_ls:
                    item.append(consume_time)
                result_list.extend(test_result_ls)
                df = pd.DataFrame(result_list,
                                  columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss', 'Comsume_time'])
            elif self.args.forget_paradigm == 'sample':
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model, self.args, test_loaders)
                for item in test_result_ls:
                    item.append(consume_time)
                result_list.extend(test_result_ls)
                df = pd.DataFrame(result_list,
                                  columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc', 'Comsume_time'])

            global_model.to('cpu')

            print("Relearn Round = {}".format(epoch))
        
        if self.args.cut_sample == 1.0:
            df.to_csv('./results/{}/relearn_data_{}_distri_{}_fnum_{}_algo_{}.csv'.format(self.args.forget_paradigm,
                                                                                          self.args.data_name,
                                                                                      self.args.alpha,
                                                                                      len(self.args.forget_class_idx),
                                                                                      self.args.paradigm), index=False)
        elif self.args.cut_sample < 1.0:
            df.to_csv('./results/{}/relearn_data_{}_distri_{}_fnum_{}_algo_{}_partdata_{}.csv'.format(self.args.forget_paradigm,
                                                                                          self.args.data_name,
                                                                                      self.args.alpha,
                                                                                      len(self.args.forget_class_idx),
                                                                                      self.args.paradigm, self.args.cut_sample), index=False)
        return