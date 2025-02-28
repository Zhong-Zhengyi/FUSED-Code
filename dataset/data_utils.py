import random

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pickle
import copy

train_size = 0.99 # merge original training set and test set, then split it manually.
least_samples = 100 # guarantee that each client must have at least one samples for testing.



def data_set(data_name):
    if (data_name == 'mnist'):
        trainset = datasets.MNIST('./dataset/mnist', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))

        testset = datasets.MNIST('./dataset/mnist', train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
    elif (data_name == 'fashionmnist'):
        trainset = datasets.MNIST('./dataset/fashionmnist', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))

        testset = datasets.MNIST('./dataset/fashionmnist', train=False, download=True,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]))
    # model: ResNet-18
    elif (data_name == 'cifar10'):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='./dataset/cifar10', train=True,
                                    download=True, transform=transform)

        testset = datasets.CIFAR10(root='./dataset/cifar10', train=False,
                                   download=True, transform=transform)

    elif (data_name == 'cifar100'):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR100('./dataset/cifar100', train=True, download=True,
                                  transform=transform)

        testset = datasets.CIFAR100('./dataset/cifar100', train=False, download=True,
                                 transform=transform)

    return trainset, testset

def separate_data(data, num_clients, num_classes, args, niid=False, balance=False, partition=None, class_per_client=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]

    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

    classes_ls = [i for i in range(num_classes)]

    if not niid:
        partition = 'pat'
        class_per_client = len(classes_ls)

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for cls in classes_ls:
            idx_for_each_class.append(idxs[dataset_label == cls])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in classes_ls:
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(np.ceil((num_clients /len(classes_ls)) *class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients -1)]
            else:
                num_samples = np.random.randint(max(num_per /10, least_samples /len(classes_ls)), num_per, num_selected_clients -1).tolist()
            num_samples.append(num_all_samples -sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx +num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = len(classes_ls)
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print \
                    (f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(args.alpha, num_clients))
                proportions = np.array([ p *(len(idx_j ) < N /num_clients) for p ,idx_j in zip(proportions ,idx_batch)])
                proportions = proportions /proportions.sum()
                proportions = (np.cumsum(proportions ) *len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j ,idx in zip(idx_batch ,np.split(idx_k ,proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client ]==i))))

    del data

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    return X, y, statistic

def split_test_proxy(test_loader, args):
    test_data_x, test_data_y, proxy_data_x, proxy_data_y = [], [], [], []
    for test_data in test_loader:
        data, label = test_data
    dataset_image = []
    dataset_label = []

    dataset_image.extend(data.cpu().detach().numpy())
    dataset_label.extend(label.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    num_classes = args.num_classes
    idxs = np.array(range(len(dataset_label)))
    idx_for_each_class = []
    for i in range(num_classes):
        idx_for_each_class.append(idxs[dataset_label == i])
        num_class_proxy = len(idx_for_each_class[i])*args.proxy_frac
        idx_class_proxy = np.random.choice(idx_for_each_class[i], int(num_class_proxy))
        idx_class_test = list(set(idx_for_each_class[i])-set(idx_class_proxy))
        proxy_data_x.extend(dataset_image[idx_class_proxy])
        proxy_data_y.extend(dataset_label[idx_class_proxy])
        test_data_x.extend(dataset_image[idx_class_test])
        test_data_y.extend(dataset_label[idx_class_test])
    proxy_data_x = np.array(proxy_data_x)
    proxy_data_y = np.array(proxy_data_y)
    test_data_x = np.array(test_data_x)
    test_data_y = np.array(test_data_y)

    X_proxy = torch.Tensor(proxy_data_x).type(torch.float32)
    y_proxy = torch.Tensor(proxy_data_y).type(torch.int64)

    data_proxy = [(x, y) for x, y in zip(X_proxy, y_proxy)]
    proxy_loader = DataLoader(data_proxy, batch_size=args.test_batch_size, shuffle=True)
    return test_data_x, test_data_y, proxy_loader


def split_proxy(x, y, args, AT=None):
    client_x, client_at, client_y, proxy_data_x, proxy_data_at, proxy_data_y = [], [], [], [], [], []

    classes_ls = [i for i in range(args.num_classes)]

    for client in range(args.num_user):
        dataset_image = x[client]
        dataset_label = y[client]
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = {}
        all_class_x = []
        all_class_y = []
        all_class_x_proxy = []
        all_class_y_proxy = []
        for i in classes_ls:
            idx_for_each_class[i] = idxs[dataset_label == i]
            num_class_proxy = len(idx_for_each_class[i])*args.proxy_frac
            idx_class_proxy = np.random.choice(idx_for_each_class[i], int(num_class_proxy))
            idx_class_client = list(set(idx_for_each_class[i])-set(idx_class_proxy))
            all_class_x_proxy.extend(dataset_image[idx_class_proxy])
            all_class_y_proxy.extend(dataset_label[idx_class_proxy])
            all_class_x.extend(dataset_image[idx_class_client])
            all_class_y.extend(dataset_label[idx_class_client])
        client_x.append(all_class_x)
        client_y.append(all_class_y)
        proxy_data_x.append(all_class_x_proxy)
        proxy_data_y.append(all_class_y_proxy)

    client_loaders, test_loaders = split_data(client_x, client_y, args)
    proxy_client_loaders, proxy_test_loaders = split_data(proxy_data_x, proxy_data_y, args)

    return client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders

def split_data(X, y, args, client_at=None):
    client_loaders, test_loaders = [], []
    if args.forget_paradigm == 'client':
        train_size = 0.7
    else:
        train_size = 0.99
    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        client_loaders.append(DataLoader(train_data, batch_size=args.local_batch_size, shuffle=True, num_workers=2,))
        test_loaders.append(DataLoader(test_data, batch_size=args.test_batch_size, shuffle=True))

    del X, y
    return client_loaders, test_loaders


