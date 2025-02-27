import sys

import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time
from utils import *
import os
import pandas as pd
import logging
from alg_utils.ada_hessain import AdaHessian
from transformers import AdamW

class Base(object):
    def __init__(self, args):
        super(Base, self).__init__()
        self.args = args
        self.log_dir = f"logs/retrain_{self.args.forget_paradigm}_{self.args.data_name}_{self.args.alpha}"
    def FL_Train(self, init_global_model, client_all_loaders, test_loader, FL_params):

        print('\n')
        print(5 * "#" + "  Federated Training Start  " + 5 * "#")
        all_global_models = list()
        all_client_models = list()
        global_model = init_global_model
        result_list = []

        all_global_models.append(global_model)

        checkpoints_ls = []
        avg_acc = 0
        for epoch in range(FL_params.global_epoch):
            last_avg_acc = avg_acc
            selected_clients = list(np.random.choice(range(FL_params.num_user), size=int(FL_params.num_user*FL_params.fraction), replace=False))
            select_client_loaders = list()
            for idx in selected_clients:
                select_client_loaders.append(client_all_loaders[idx])

            client_models = self.global_train_once(epoch, global_model, select_client_loaders, test_loader, FL_params, checkpoints_ls)

            all_client_models += client_models
            global_model = self.fedavg(client_models)
            all_global_models.append(copy.deepcopy(global_model).to('cpu'))

            all_idx = [k for k in range(FL_params.num_user)]
            client_test_acc = []

            for client_idx in all_idx:
                (test_loss, test_acc) = self.test(global_model, test_loader[client_idx], FL_params)
                client_test_acc.append(test_acc)
                result_list.append([epoch, client_idx, test_loss, test_acc])
            global_model.to('cpu')
            avg_acc = sum(client_test_acc) / len(client_test_acc)#TODO: 这里假设每个客户端的数据量是相同的

            print("FL Round = {}, Global Model Accuracy= {}".format(epoch, avg_acc))

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Test_acc', 'Test_loss'])
        if self.args.save_normal_result:
            df.to_csv('./results/Acc_loss_fl_data_{}_distri_{}.csv'.format(FL_params.data_name, FL_params.alpha))

        print(5 * "#" + "  Federated Training End  " + 5 * "#")

        return all_global_models, all_client_models


    def FL_Retrain(self, init_global_model,  client_all_loaders, test_loaders, FL_params):
        if (FL_params.if_retrain == False):
            raise ValueError('FL_params.if_retrain should be set to True, if you want to retrain FL model')

        print('\n')
        print(5 * "#" + "  Federated Retraining Start  " + 5 * "#")
        std_time = time.time()
        # retrain_GMs = list()

        # retrain_GMs.append(copy.deepcopy(init_global_model))
        global_model = copy.deepcopy(init_global_model)
        checkpoints_ls = []
        # gap = 0
        result_list = []
        for epoch in range(FL_params.global_epoch):
            # last_gap = gap
            selected_clients = list(np.random.choice(range(FL_params.num_user), size=int(FL_params.num_user * FL_params.fraction), replace=False))
            if FL_params.forget_paradigm == 'client': # 将需要遗忘的客户端排除在外
                selected_clients = [value for value in selected_clients if value not in FL_params.forget_client_idx]

            self.select_forget_idx = list()
            # select_client_loaders = list()
            record = -1
            for idx in selected_clients:
                # select_client_loaders.append(client_all_loaders[idx])
                record += 1
                if idx in self.args.forget_client_idx:
                    self.select_forget_idx.append(record)
            select_client_loaders = select_part_sample(self.args, client_all_loaders, selected_clients)

            client_models = self.global_train_once(epoch, global_model,  select_client_loaders, test_loaders, FL_params, checkpoints_ls)
            global_model = self.fedavg(client_models)
            # global_model_ls = [copy.deepcopy(global_model) for _ in range(FL_params.num_user)]
            if self.args.forget_paradigm == 'client':
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, global_model, self.args, test_loaders)
                print('Epoch {}, Remember Test Acc={}, Forget Test Acc={}'.format(epoch, avg_r_acc, avg_f_acc))
            elif self.args.forget_paradigm == 'class':
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args,
                                                                          test_loaders)
                print('Epoch {}, Remember Test Acc={}, Forget Test Acc={}'.format(epoch, avg_r_acc, avg_f_acc))
            elif self.args.forget_paradigm == 'sample':
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model,
                                                                                              self.args, test_loaders)
                print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero,
                                                                                 avg_test_acc))
            result_list.extend(test_result_ls)
            # gap = avg_r_acc - avg_f_acc

            # retrain_GMs.append(copy.deepcopy(global_model))

        end_time = time.time()
        consume_time = end_time - std_time
        if FL_params.forget_paradigm == 'client':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss'])
        elif FL_params.forget_paradigm == 'class':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss'])
        elif FL_params.forget_paradigm == 'sample':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc'])
        df['Comsume_time'] = consume_time

        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                './results/{}/Acc_loss_retrain_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm, self.args.forget_paradigm, self.args.data_name, self.args.alpha, len(self.args.forget_class_idx)))
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/Acc_loss_retrain_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'.format(self.args.forget_paradigm,
                                                                                     self.args.forget_paradigm,
                                                                                     self.args.data_name,
                                                                                     self.args.alpha,
                                                                                     len(self.args.forget_class_idx), self.args.cut_sample))

        print(5 * "#" + "  Federated Retraining End  " + 5 * "#")
        return global_model

    """
    Function：
    For the global round of training, the data and optimizer of each global_ModelT is used. The global model of the previous round is the initial point and the training begins.
    NOTE:The global model inputed is the global model for the previous round
        The output client_Models is the model that each user trained separately.
    """

    # training sub function
    def global_train_once(self, epoch, global_model, client_data_loaders, test_loaders, FL_params, checkpoints_ls):
        global_model.to(FL_params.device)
        device_cpu = torch.device("cpu")
        client_models = []
        lr = FL_params.lr

        # if FL_params.paradigm == 'federaser':
        #     for ii in range(len(client_data_loaders)):
        #         client_models.append("1")
        # else:
        # for ii in range(len(client_data_loaders)):
        #     client_models.append(copy.deepcopy(global_model))

        for idx, client_data in enumerate(client_data_loaders):
            model = copy.deepcopy(global_model)
            if self.args.data_name == 'text':
                optimizer = AdamW(model.parameters(), lr=1e-5)
            else:
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            # model.to(device)
            model.train()

            # local training
            if self.args.paradigm == 'infocom22' and self.args.if_unlearning == True:
                self.local_train_infocom22(model, optimizer, client_data_loaders[idx], FL_params)
            else:
                model = self.local_train(model, optimizer, client_data_loaders[idx], FL_params)
            

            client_models.append(model)

            if self.args.paradigm == 'lora':
                for name, param in model.named_parameters():
                    for name_, param_ in global_model.named_parameters():
                        if name == name_:
                            pdist = nn.PairwiseDistance(p=1)
                            param_size = sys.getsizeof(param.data)
                            diff = pdist(param.data, param_.data)
                            diff = torch.norm(diff)
                            self.param_change_dict[name] = diff
                            self.param_size[name] = param_size
            model.to(device_cpu)
        return client_models


    """
    Function：
    Test the performance of the model on the test set
    """
    def local_train(self, model, optimizer, data_loader, FL_params):
        for local_epoch in range(FL_params.local_epoch):
            criteria = nn.CrossEntropyLoss()
            if self.args.data_name == 'text':
                for batch in data_loader:
                    optimizer.zero_grad()
                    input_ids = batch[0].to(FL_params.device)
                    input_ids = input_ids.long()
                    input_ids = input_ids.to(FL_params.device)
                    attention_mask = batch[1].to(FL_params.device)
                    labels = batch[2].to(FL_params.device)
                    outputs = model(input_ids, attention_mask)
                    logits = outputs.logits
                    loss = criteria(logits, labels)
                    loss.backward()
                    optimizer.step()
            else:
                for batch_idx, (data, target) in enumerate(data_loader):
                    optimizer.zero_grad()
                    data = data.to(FL_params.device)
                    target = target.to(FL_params.device)
                    pred = model(data)

                    loss = criteria(pred, target)
                    loss.backward()
                    optimizer.step()
        return model

    def local_train_infocom22(self, model, optimizer, data_loader, FL_params):
        optimizer = AdaHessian(model.parameters())
        for local_epoch in range(FL_params.local_epoch):
            for batch_idx, (data, target) in enumerate(data_loader):
                model.zero_grad()
                data = data.to(FL_params.device)
                target = target.to(FL_params.device)

                pred, _ = model(data)
                criteria = nn.CrossEntropyLoss()
                loss = criteria(pred, target)
                loss.backward(create_graph=True)
                optimizer.step()
                optimizer.zero_grad()




    def test(self, model, test_loader,FL_params):
        for param in model.parameters():
            device = param.device
            break
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        true_zero_total = 0
        acc_true_zero_total = 0
        pred_zero_total = 0
        jingdu_pred_zero_total = 0
        criteria = nn.CrossEntropyLoss()
        with torch.no_grad():
            if self.args.data_name == 'text':
                for batch in test_loader:
                    model.to(FL_params.device)
                    input_ids = batch[0].to(FL_params.device)
                    input_ids = input_ids.long()
                    input_ids = input_ids.to(FL_params.device)
                    attention_mask = batch[1].to(FL_params.device)
                    target = batch[2].to(FL_params.device)
                    outputs = model(input_ids, attention_mask)
                    logits = outputs.logits
                    test_loss += criteria(logits, target)
                    pred = torch.argmax(logits, axis=1)
                    pred = pred.cpu()
                    target = target.cpu()
                    correct += torch.sum(torch.eq(pred, target)).item()
                    total += len(target)
            else:
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    model.to(device)
                    output = model(data)
                    test_loss += criteria(output, target)  # sum up batch loss
                    pred = torch.argmax(output, axis=1)
                    pred = pred.cpu()
                    target = target.cpu()
                    if FL_params.forget_paradigm == 'sample':
                        pred_zero_indices = np.where(pred == 0)
                        pred_zero_count = np.count_nonzero(pred == 0)
                        pred_zero_total += pred_zero_count
                        for ele in target[pred_zero_indices]:
                            if ele == 0:
                                jingdu_pred_zero_total += 1
                        true_zero_indices = np.where(target == 0)
                        for ele1 in pred[true_zero_indices]:
                            if ele1 == 0:
                                acc_true_zero_total += 1
                        true_zero_count = np.count_nonzero(target == 0)
                        true_zero_total += true_zero_count
                        correct += torch.sum(torch.eq(pred, target)).item()
                        total += len(target)
                    else:
                        correct += torch.sum(torch.eq(pred, target)).item()
                        total += len(target)
        if FL_params.forget_paradigm == 'sample':
            if pred_zero_total == 0:
                jingdu = np.nan
            else:
                jingdu = jingdu_pred_zero_total/pred_zero_total
            if true_zero_total == 0:
                acc_zero = np.nan
            else:
                acc_zero = acc_true_zero_total/true_zero_total
            test_acc = correct/total
            return (jingdu, acc_zero, test_acc)
        else:
            test_loss /= len(test_loader.dataset)
            test_acc = correct/total
            return (test_loss, test_acc)


    """
    Function：
    FedAvg
    """
    def fedavg(self, local_models):
        """
        Parameters
        ----------
        local_models : list of local models
            DESCRIPTION.In federated learning, with the global_model as the initial model, each user uses a collection of local models updated with their local data.
        local_model_weights : tensor or array
            DESCRIPTION. The weight of each local model is usually related to the accuracy rate and number of data of the local model.(Bypass)

        Returns
        -------
        update_global_model
            Updated global model using fedavg algorithm
        """

        global_model = copy.deepcopy(local_models[0])

        avg_state_dict = global_model.state_dict()
        local_state_dicts = list()
        for model in local_models:
            local_state_dicts.append(model.state_dict())

        for layer in avg_state_dict.keys():
            avg_state_dict[layer] = 0
            # for client_idx in range(len(local_models)):
            #     avg_state_dict[layer] += local_state_dicts[client_idx][layer]*self.args.datasize_ls[client_idx]
            # if 'num_batches_tracked' in layer:
            #     avg_state_dict[layer] = (avg_state_dict[layer]/sum(self.args.datasize_ls)).long()
            # else:
            #     avg_state_dict[layer] /= sum(self.args.datasize_ls)
            for client_idx in range(len(local_models)):
                avg_state_dict[layer] += local_state_dicts[client_idx][layer]
            if 'num_batches_tracked' in layer:
                avg_state_dict[layer] = (avg_state_dict[layer] / len(local_models)).long()
            else:
                avg_state_dict[layer] /= len(local_models)

        global_model.load_state_dict(avg_state_dict)

        return global_model

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
