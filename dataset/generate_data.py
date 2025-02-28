import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from dataset.data_utils import data_set, separate_data, split_proxy
import numpy as np
import os
import pickle as pkl
import pandas as pd
import tqdm
import random
from transformers import BertTokenizer

MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'

# Allocate data to users


def data_init(FL_params):
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.device == 'cuda' else {}
    dataset_x = []
    dataset_at = []
    dataset_y = []

    trainset, testset = data_set(FL_params.data_name)

    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, num_workers=2, **kwargs)
    train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, num_workers=2, **kwargs)

    for train_data in train_loader:
        x_train, y_train = train_data
        dataset_x.extend(x_train.cpu().detach().numpy())
        dataset_y.extend(y_train.cpu().detach().numpy())
    if FL_params.forget_paradigm == 'client':
        for test_data in test_loader:
            x_test, y_test = test_data
            dataset_x.extend(x_test.cpu().detach().numpy())
            dataset_y.extend(y_test.cpu().detach().numpy())

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)

    X, y, statistic = separate_data((dataset_x, dataset_y), FL_params.num_user, FL_params.num_classes, FL_params,
                                    FL_params.niid, FL_params.balance, FL_params.partition, class_per_client=2)

    client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders = split_proxy(X, y, FL_params)
    FL_params.datasize_ls = [len(k) for k in X]
    if FL_params.forget_paradigm == 'client':
        test_loaders = test_loaders
        proxy_test_loaders = proxy_test_loaders
    else:
        proxy_test_x = []
        proxy_test_y = []
        for i in range(FL_params.num_user):
            for x, y in test_loaders[i]:
                proxy_test_x.append(x)
                proxy_test_y.append(y)
        proxy_test_x = torch.cat(proxy_test_x).numpy()
        proxy_test_y = torch.cat(proxy_test_y).numpy()
        proxy_test_loader = DataLoader(TensorDataset(torch.tensor(proxy_test_x), torch.tensor(proxy_test_y)), batch_size=FL_params.test_batch_size, shuffle=True)
        proxy_test_loaders = [proxy_test_loader for _ in range(FL_params.num_user)]
    return client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders

def cross_data_init(FL_params):
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.device == 'cuda' else {}
    dataset_x = []
    dataset_y = []

    trainset, testset = data_set(FL_params.data_name)

    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, num_workers=2, **kwargs)
    train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, num_workers=2,
                              **kwargs)

    for train_data in train_loader:
        x_train, y_train = train_data
        dataset_x.extend(x_train.cpu().detach().numpy())
        dataset_y.extend(y_train.cpu().detach().numpy())
    if FL_params.forget_paradigm == 'client':
        for test_data in test_loader:
            x_test, y_test = test_data
            dataset_x.extend(x_test.cpu().detach().numpy())
            dataset_y.extend(y_test.cpu().detach().numpy())

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)

    class_num = int(FL_params.num_classes/FL_params.num_user)
    X = []
    y = []
    idx_ls = []
    for user in range(FL_params.num_user):
        idx = []
        for i in range(class_num):
            item = user*class_num + i
            indices = [idx for idx, label in enumerate(dataset_y) if label == item]
            idx.extend(indices)
        idx_ls.append(idx)
    corss_idx = idx_ls[0][:int(len(idx_ls[0])*0.01)]
    idx_ls[0] = idx_ls[0][int(len(idx_ls[0])*0.01):]
    idx_ls[1] = corss_idx + idx_ls[1]
    remain_idx = []
    for idx in range(1, FL_params.num_user):
        remain_idx.extend(idx_ls[idx])
    random.shuffle(remain_idx)
    sublist_size = len(remain_idx) // (FL_params.num_user-len(FL_params.forget_client_idx))
    remainder = len(remain_idx) % (FL_params.num_user-len(FL_params.forget_client_idx))

    sublists = [remain_idx[i * sublist_size + min(i, remainder):(i + 1) * sublist_size + min(i + 1, remainder)] for i in
                range(9)]

    for idx in range(1, FL_params.num_user):
        idx_ls[idx] = sublists[idx-1]

    for user in range(FL_params.num_user):
        X.append(dataset_x[idx_ls[user]])
        y.append(dataset_y[idx_ls[user]])

    for i in range(FL_params.num_user):
        print('client {} data size {} lable {}'.format(i, len(X[i]),np.unique(y[i])))

    client_loaders, test_loaders, proxy_loader = split_proxy(X, y, FL_params)
    FL_params.datasize_ls = [len(k) for k in X]
    if FL_params.forget_paradigm == 'client':
        test_loaders = test_loaders
    else:
        test_loaders = [test_loader for _ in range(FL_params.num_user)]

    return client_loaders, test_loaders, proxy_loader
