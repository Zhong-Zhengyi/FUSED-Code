import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from models import LeNet_FashionMNIST, CNN_Cifar10, CNN_Cifar100, ViT_Cifar100
import numpy as np
import torch.nn.functional as F
import copy
from transformers import AdamW
from torch import optim
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import random
import time

def model_init(args):
    if args.data_name == 'mnist':
        args.num_classes = 10
        args.lr = 0.01
        args.midimension = 84
        model = LeNet_FashionMNIST.Model(args)
        init_network(model)
    elif args.data_name == 'fashionmnist':
        args.num_classes = 10
        args.lr = 0.005
        # args.midimension = 84
        model = LeNet_FashionMNIST.Model(args)
        init_network(model)
    elif args.data_name == 'cifar10':
        args.num_classes = 10
        args.lr = 0.005
        # args.midimension = 512
        model = CNN_Cifar10.Model(args)
#         model = CNN_Cifar10.Model(
#     image_size = 32,
#     patch_size = 4,
#     num_classes = 10,
#     dim = 512,
#     depth = 6,
#     heads = 8,
#     mlp_dim = 512
# )
        init_network(model)
    elif args.data_name == 'cifar100':
        args.num_classes = 100
        args.lr = 0.01
        # args.midimension = 512
        model = CNN_Cifar100.Model(args)
        init_network(model)
    return model

def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        # print('Name: ', name)
        if exclude in name:
            continue
        if 'bias' in name:
            nn.init.constant_(w, 0)
        else:
            if 'batch' in name:
                nn.init.normal_(w)
                continue
            if method == 'xavier':
                if len(w.shape) < 2:
                    nn.init.kaiming_normal_(w.unsqueeze(0))
                else:
                    nn.init.kaiming_normal_(w)
                # nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)

def save_checkpoint(state, is_best, args, epoch, filename='checkpoint.pth'):
    if is_best == False:
        filename = 'checkpoint-{}-{}.pth'.format(args.dataset, epoch)
    elif is_best == True:
        filename = 'bestcheckpoint-{}.pth'.format(args.dataset)
    torch.save(state, filename)

def load_checkpoint(filename):
    model, optimizer = torch.load(filename)
    return model, optimizer

def test_class_forget(obj, epoch, model, args, test_loaders):
    forget_acc_ls = []
    remember_acc_ls = []
    test_result_ls = []
    for k in range(args.num_user):
        test_loader = test_loaders[k]
        label_data_dict = {}
        for data, target in test_loader:
            data = data.tolist()
            targets = target.tolist()
            for idx, label in enumerate(targets):
                if label not in label_data_dict:
                    label_data_dict[label] = []
                label_data_dict[label].append((torch.tensor(data[idx]), torch.tensor(label)))
        label_data_loaders = {}
        for label, data_list in label_data_dict.items():
            # class_dataset = Dataset(data_list)
            class_loader = DataLoader(data_list, batch_size=args.test_batch_size, shuffle=True)
            label_data_loaders[label] = class_loader

        for label, loader in label_data_loaders.items():
            (test_loss, test_acc) = obj.test(model, label_data_loaders[label], args)
            test_result_ls.append([epoch, k, label, test_acc, float(test_loss)])
            if label in args.forget_class_idx:
                forget_acc_ls.append(test_acc)
            else:
                remember_acc_ls.append(test_acc)
    avg_f_acc = sum(forget_acc_ls) / len(forget_acc_ls)
    avg_r_acc = sum(remember_acc_ls) / len(remember_acc_ls)

    return avg_f_acc, avg_r_acc, test_result_ls

def test_backdoor_forget(obj, epoch, model, args, test_loaders):
    jingdu_ls = []
    acc_zero_ls = []
    test_acc_ls = []
    test_result_ls = []
    for k in range(args.num_user):
        test_loader = test_loaders[k]

        dataset_x = []
        dataset_y = []
        for data, target in test_loader:
            x_bk = copy.deepcopy(data)
            x_bk[:, :, -1, -1] = 255
            # data, target = insert_backdoor(args, data, target)
            dataset_x.extend(data.cpu().detach().numpy())
            dataset_y.extend(target.cpu().detach().numpy())
        dataset_x = np.array(dataset_x)
        dataset_y = np.array(dataset_y)
        dataset_x = torch.Tensor(dataset_x).type(torch.float32)
        dataset_y = torch.Tensor(dataset_y).type(torch.int64)
        dataset = [(x, y) for x, y in zip(dataset_x, dataset_y)]

        test_data = torch.utils.data.DataLoader(dataset, batch_size=args.test_batch_size,
                                                                 shuffle=True)
        (jingdu, acc_zero, test_acc) = obj.test(model, test_data, args)
        test_result_ls.append([epoch, k, jingdu, acc_zero, test_acc])
        jingdu_ls.append(jingdu)
        acc_zero_ls.append(acc_zero)
        test_acc_ls.append(test_acc)

    avg_jingdu = sum(jingdu_ls) / len(jingdu_ls)
    avg_acc_zero = sum(acc_zero_ls) / len(acc_zero_ls)
    avg_test_acc = sum(test_acc_ls) / len(test_acc_ls)

    return avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls

def test_client_forget(obj, epoch, model, args, test_loaders):
    all_idx = [k for k in range(args.num_user)]
    forget_acc_ls = []
    remember_acc_ls = []
    test_result_ls = []
    for k in range(args.num_user):
        test_loader = test_loaders[k]
        label_data_dict = {}
        for data, target in test_loader:
            data = data.tolist()
            targets = target.tolist()
            for idx, label in enumerate(targets):
                if label not in label_data_dict:
                    label_data_dict[label] = []
                label_data_dict[label].append((torch.tensor(data[idx]), torch.tensor(label)))
        label_data_loaders = {}
        for label, data_list in label_data_dict.items():
            class_loader = DataLoader(data_list, batch_size=len(data_list), shuffle=True)
            label_data_loaders[label] = class_loader

        for label, loader in label_data_loaders.items():
            (test_loss, test_acc) = obj.test(model, label_data_loaders[label], args)
            label_num = 0
            for data, target in label_data_loaders[label]:
                label_num += data.size(0)
            test_result_ls.append([epoch, k, label, label_num, test_acc, float(test_loss)])
            if k in args.forget_client_idx:
                forget_acc_ls.append(test_acc)
            else:
                remember_acc_ls.append(test_acc)
    avg_f_acc = sum(forget_acc_ls) / len(forget_acc_ls)
    avg_r_acc = sum(remember_acc_ls) / len(remember_acc_ls)
    return avg_f_acc, avg_r_acc, test_result_ls

def erase_forget_class(args, client_loaders):
    for user in range(args.num_user):
        dataset_image = []
        dataset_at = []
        dataset_label = []

        for x, y in client_loaders[user]:
            dataset_image.extend(x)
            dataset_label.extend(y)
        data_x = []
        data_y = []
        for idx, cls in enumerate(dataset_label):
            if cls not in args.forget_class_idx:
                data_x.append(dataset_image[idx].cpu().detach().numpy())
                data_y.append(dataset_label[idx].cpu().detach().numpy())

        data_x = np.array(data_x)
        data_y = np.array(data_y)
        data_x = torch.Tensor(data_x).type(torch.float32)
        data_y = torch.Tensor(data_y).type(torch.int64)
        train_data = [(x1, y1) for x1, y1 in zip(data_x, data_y)]
        client_loaders[user] = torch.utils.data.DataLoader(train_data, batch_size=args.local_batch_size, shuffle=True)
    return client_loaders

def select_forget_sample(args, client_loaders):
    for user in range(args.num_user):
        dataset_x = []
        dataset_y = []
        for idx, (data, target) in enumerate(client_loaders[user]):
            if (user in args.forget_client_idx) and (idx <= 1):  # 前五个batch植入后门
                dataset_x.extend(data)
                dataset_y.extend(target)
        if len(dataset_x) == 0:
            train_data = None
            client_loaders[user] = None
        else:
            dataset_x = np.array(dataset_x)
            dataset_y = np.array(dataset_y)
            dataset_x = torch.Tensor(dataset_x).type(torch.float32)
            dataset_y = torch.Tensor(dataset_y).type(torch.int64)
            dataset = [(x, y) for x, y in zip(dataset_x, dataset_y)]
            client_loaders[user] = torch.utils.data.DataLoader(dataset, batch_size=args.local_batch_size,
                                                                shuffle=False)
    return client_loaders

def select_forget_class(args, client_loaders):
    for user in range(args.num_user):
        dataset_image = []
        dataset_at = []
        dataset_label = []
        for x, y in client_loaders[user]:
            dataset_image.extend(x)
            dataset_label.extend(y)
        data_x = []
        data_y = []
        for idx, cls in enumerate(dataset_label):
            if cls in args.forget_class_idx:
                data_x.append(dataset_image[idx].cpu().detach().numpy())
                data_y.append(dataset_label[idx].cpu().detach().numpy())

        if len(data_x) == 0:
            train_data = None
            client_loaders[user] = None
        else:
            data_x = np.array(data_x)
            data_y = np.array(data_y)
            data_x = torch.Tensor(data_x).type(torch.float32)
            data_y = torch.Tensor(data_y).type(torch.int64)
            train_data = [(x1, y1) for x1, y1 in zip(data_x, data_y)]
            client_loaders[user] = torch.utils.data.DataLoader(train_data, batch_size=args.local_batch_size, shuffle=True)
    return client_loaders

def baizhanting_attack(args, client_loaders, test_loaders):
    for user in args.forget_client_idx:
        dataset_image = []
        dataset_label = []
        test_image = []
        test_label = []
        for x, y in client_loaders[user]:
            dataset_image.extend(x)
            dataset_label.extend(y)
        for x, y in test_loaders[user]:
            test_image.extend(x)
            test_label.extend(y)
        data_x = []
        data_y = []
        test_x = []
        test_y = []
        for idx, cls in enumerate(dataset_label):
            if int(dataset_label[idx]) < args.num_classes-1:
                dataset_label[idx] = int(dataset_label[idx])+1
            elif int(dataset_label[idx])==args.num_classes-1:
                dataset_label[idx] = 0
            data_x.append(np.array(dataset_image[idx]))
            data_y.append(np.array(dataset_label[idx]))

        data_x = np.array(data_x)
        data_y = np.array(data_y)
        data_x = torch.Tensor(data_x).type(torch.float32)
        data_y = torch.Tensor(data_y).type(torch.int64)
        train_data = [(x1, y1) for x1, y1 in zip(data_x, data_y)]
        client_loaders[user] = torch.utils.data.DataLoader(train_data, batch_size=args.local_batch_size,
                                                           shuffle=True)
        for idx, cls in enumerate(test_label):
            if int(test_label[idx]) < args.num_classes-1:
                test_label[idx] = int(test_label[idx])+1
            elif int(test_label[idx])==args.num_classes-1:
                test_label[idx] = 0
            test_x.append(np.array(test_image[idx]))
            test_y.append(np.array(test_label[idx]))

        test_x = np.array(test_x)
        test_y = np.array(test_y)
        test_x = torch.Tensor(test_x).type(torch.float32)
        test_y = torch.Tensor(test_y).type(torch.int64)
        test_data = [(x1, y1) for x1, y1 in zip(test_x, test_y)]
        test_loaders[user] = torch.utils.data.DataLoader(test_data, batch_size=args.local_batch_size,
                                                           shuffle=True)
    return client_loaders, test_loaders

def backdoor_attack(args, client_loaders):
    for client in args.forget_client_idx:
        dataset_x = []
        dataset_y = []
        for idx, (data, target) in enumerate(client_loaders[client]):
            if idx <= 1:
                data_, target_ = insert_backdoor(args, data, target)
                dataset_x.extend(data_.cpu().detach().numpy())
                dataset_y.extend(target_.cpu().detach().numpy())
            else:
                dataset_x.extend(data.cpu().detach().numpy())
                dataset_y.extend(target.cpu().detach().numpy())
        dataset_x = np.array(dataset_x)
        dataset_y = np.array(dataset_y)
        dataset_x = torch.Tensor(dataset_x).type(torch.float32)
        dataset_y = torch.Tensor(dataset_y).type(torch.int64)
        dataset = [(x, y) for x, y in zip(dataset_x, dataset_y)]
        client_loaders[client] = torch.utils.data.DataLoader(dataset, batch_size=args.local_batch_size,
                                                             shuffle=False)
    return client_loaders

def erase_backdoor(args, client_loaders):
    for client in args.forget_client_idx:
        dataset_x = []
        dataset_y = []
        for idx, (data, target) in enumerate(client_loaders[client]):
            if idx <= 1:
                continue
            dataset_x.extend(data.cpu().detach().numpy())
            dataset_y.extend(target.cpu().detach().numpy())
        dataset_x = np.array(dataset_x)
        dataset_y = np.array(dataset_y)
        dataset_x = torch.Tensor(dataset_x).type(torch.float32)
        dataset_y = torch.Tensor(dataset_y).type(torch.int64)
        dataset = [(x, y) for x, y in zip(dataset_x, dataset_y)]
        client_loaders[client] = torch.utils.data.DataLoader(dataset, batch_size=args.local_batch_size,
                                                             shuffle=False)
    return client_loaders

def insert_backdoor(args, data, target, trigger_label=0, trigger_pixel_value=255):
    x_bk = copy.deepcopy(data)
    x_bk[:, :, -1, -1] = trigger_pixel_value
    y_bk = torch.full_like(target, trigger_label)
    return x_bk, y_bk


    
# construct attack model
class FCNet(nn.Module):

    def __init__(self, args, dim_hidden = 20, dim_out = 2):
        super(FCNet, self).__init__()
       
        self.fc1 = nn.Linear(args.num_classes, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_out)

        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train(model, optimizer, data_loader, test_loaders, epochs, args):
    criteria = nn.CrossEntropyLoss()
    result_ls = []
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            target = target.to(args.device).long()
            pred = model(data)
            loss = criteria(pred, target)
            loss.backward()
            optimizer.step()
        # evaluation
        model.eval()
        if args.forget_paradigm == 'class':
            for k,v in test_loaders.items():
                testloader = v
                num = 0
                with torch.no_grad():
                    c_pred_y, c_true_y = [], []
                    for data in testloader:
                        images, labels = data[0].to(args.device), data[1].to(args.device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c_pred_y.append(predicted.cpu().numpy())
                        c_true_y.append(labels.cpu().numpy())
                        num += labels.size(0)
                    c_true_y, c_pred_y = np.concatenate(c_true_y), np.concatenate(c_pred_y)
               
                print("Accuracy score for class %d:"%k)
                print(accuracy_score(c_true_y, c_pred_y))
                result_ls.append([epoch, k, accuracy_score(c_true_y, c_pred_y), num])
                
        else:
            for client in range(args.num_user):
                testloader = test_loaders[client]
                num = 0
                with torch.no_grad():
                    c_pred_y, c_true_y = [], []
                    for data in testloader:
                        images, labels = data[0].to(args.device), data[1].to(args.device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        c_pred_y.append(predicted.cpu().numpy())
                        c_true_y.append(labels.cpu().numpy())
                        num += labels.size(0)
                    c_true_y, c_pred_y = np.concatenate(c_true_y), np.concatenate(c_pred_y)

                print("Accuracy score for client %d:"%client)
                print(accuracy_score(c_true_y, c_pred_y))
                result_ls.append([epoch, client, accuracy_score(c_true_y, c_pred_y), num])
                
    return model, result_ls

def reduce_ones(x, y, classes):
    ## assumes more training than testing examples
    ## 1 as over-represented class is hardcoded in here
    
    idx_to_keep = np.where(y == 0)[0]
    idx_to_reduce = np.where(y == 1)[0]
    num_to_reduce = (y.shape[0] - idx_to_reduce.shape[0]) * 2
    num_to_reduce = min(num_to_reduce, idx_to_reduce.shape[0]) 
    idx_sample = np.random.choice(idx_to_reduce, num_to_reduce, replace = False)

    x = x[np.concatenate([idx_to_keep, idx_sample, idx_to_keep])]
    y = y[np.concatenate([idx_to_keep, idx_sample, idx_to_keep])]
    classes = classes[np.concatenate([idx_to_keep, idx_sample, idx_to_keep])]

    return x, y, classes

def membership_inference_attack(args, unlearning_model, case, model, client_all_loaders_bk, test_loaders, proxy_client_loaders_bk, proxy_client_loaders, proxy_test_loaders):
    test_x, test_y, test_classes = [], [], []
    test_x_user, test_y_user = {}, {}
    for i in range(args.num_user):
        test_x_user[i] = []
        test_y_user[i] = []
    for i in range(args.num_user):
        data_loader = client_all_loaders_bk[i]
        for batch_idx, data in enumerate(data_loader):
            unlearning_model.to(args.device)
            images, labels = data[0].to(args.device), data[1].to(args.device)
            outputs = unlearning_model(images)
            test_x.extend(outputs.cpu().detach())
            test_x_user[i].extend(outputs.cpu().detach())
            labels = labels.cpu()
            if args.forget_paradigm == 'class':
                matrix = np.ones(len(labels))
                unlearn_idx = np.where(np.isin(labels, args.forget_class_idx))[0]
                for idx in unlearn_idx:
                    matrix[idx] = 0
                test_y.extend(matrix)
                test_y_user[i].extend(matrix)
            elif args.forget_paradigm == 'client':
                if i in args.forget_client_idx:
                    test_y.extend(np.zeros(len(labels)))
                    test_y_user[i].extend(np.zeros(len(labels)))
                else:
                    test_y.extend(np.ones(len(labels)))
                    test_y_user[i].extend(np.ones(len(labels)))
            elif args.forget_paradigm == 'sample':
                if (i in args.forget_client_idx) and (batch_idx <= 1):
                    test_y.extend(np.zeros(len(labels)))
                    test_y_user[i].extend(np.zeros(len(labels)))
                else:
                    test_y.extend(np.ones(len(labels)))
                    test_y_user[i].extend(np.ones(len(labels)))
            test_classes.extend(labels)
   
    if args.forget_paradigm == 'class':
        test_loader = test_loaders[0]
        for data in test_loader:
            images, labels = data[0].to(args.device), data[1].to(args.device)
            outputs = unlearning_model(images)
            test_x.extend(outputs.cpu().detach())
            labels = labels.cpu()
            matrix = np.ones(len(labels))
            unlearn_idx = np.where(np.isin(labels, args.forget_class_idx))[0]
            for idx in unlearn_idx:
                matrix[idx] = 0
            test_y.extend(matrix)
            test_classes.extend(labels)
    elif args.forget_paradigm == 'client':
        for i in range(args.num_user):
            test_loader = test_loaders[i]
            for data in test_loader:
                images, labels = data[0].to(args.device), data[1].to(args.device)
                outputs = unlearning_model(images)
                test_x.extend(outputs.cpu().detach())
                test_x_user[i].extend(outputs.cpu().detach())
                labels = labels.cpu()
                if i in args.forget_client_idx:
                    test_y.extend(np.zeros(len(labels)))
                    test_y_user[i].extend(np.zeros(len(labels)))
                else:
                    test_y.extend(np.ones(len(labels)))
                    test_y_user[i].extend(np.ones(len(labels)))
                test_classes.extend(labels)
    elif args.forget_paradigm == 'sample':
        for i in range(args.num_user):
            test_loader = test_loaders[i]
            for batch_idx, data in enumerate(test_loader):
                images, labels = data[0].to(args.device), data[1].to(args.device)
                outputs = unlearning_model(images)
                test_x.extend(outputs.cpu().detach())
                test_x_user[i].extend(outputs.cpu().detach())
                labels = labels.cpu()
                test_y.extend(np.ones(len(labels)))
                test_y_user[i].extend(np.ones(len(labels)))
                test_classes.extend(labels)

    test_x = np.array(test_x)
    test_y = np.array(test_y)
    test_classes = np.array(test_classes)
    test_x = test_x.astype('float32')
    test_y = test_y.astype('int32')
    test_classes = test_classes.astype('int32')
    for k, v in test_x_user.items():
        test_x_user[k] = np.array(v)
        test_x_user[k] = test_x_user[k].astype('float32')
        test_y_user[k] = np.array(test_y_user[k])
        test_y_user[k] = test_y_user[k].astype('int32')
    
    # training shadow model using proxy data
    attack_x_train, attack_y_train, classes_train,attack_x_train_user, attack_y_train_user = train_shadow_model(args, case, model, proxy_client_loaders_bk, proxy_client_loaders, proxy_test_loaders)

    ## balance datasets
    attack_x_train, attack_y_train, classes_train = reduce_ones(attack_x_train, attack_y_train, classes_train)
    test_x, test_y, test_classes = reduce_ones(test_x, test_y, test_classes)

    # training attack model using proxy data
    train_indices = np.arange(len(attack_x_train))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(test_classes)
    trainloader = DataLoader(TensorDataset(torch.tensor(attack_x_train), torch.tensor(attack_y_train)), batch_size=args.test_batch_size, shuffle=True)
    if args.forget_paradigm == 'class':
        test_loaders = {}
        for c in unique_classes:
            c_test_indices = test_indices[test_classes == c]
            c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]
            loader = DataLoader(TensorDataset(torch.tensor(c_test_x), torch.tensor(c_test_y)), batch_size=args.test_batch_size, shuffle=True)
            test_loaders[c] = loader
    elif args.forget_paradigm == 'client':
        test_loaders = {}
        for client in range(args.num_user):
            testloader = DataLoader(TensorDataset(torch.tensor(test_x_user[client]), torch.tensor(test_y_user[client])), batch_size=args.test_batch_size, shuffle=True)
            test_loaders[client] = testloader
    elif args.forget_paradigm == 'sample':
        test_loaders = {}
        for client in range(args.num_user):
            testloader = DataLoader(TensorDataset(torch.tensor(test_x_user[client]), torch.tensor(test_y_user[client])), batch_size=args.test_batch_size, shuffle=True)
            test_loaders[client] = testloader
    attack_model = FCNet(args=args)

    optimizer = optim.SGD(attack_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    attack_model.to(args.device)
    attack_model.train()
    output_layer, result_ls = train(attack_model, optimizer, trainloader, test_loaders, args.global_epoch, args)
        
    if args.forget_paradigm == 'class':
        df = pd.DataFrame(result_ls, columns=['Epoch', 'Class', 'Accuracy', 'Number'])
    elif args.forget_paradigm == 'client':
        df = pd.DataFrame(result_ls, columns=['Epoch', 'Client', 'Accuracy', 'Number'])
    elif args.forget_paradigm == 'sample':
        df = pd.DataFrame(result_ls, columns=['Epoch', 'Client', 'Accuracy', 'Number'])
    if args.cut_sample == 1.0:
        df.to_csv(
            './results/{}/MIA_{}_data_{}_distri_{}_fnum_{}_algo_{}.csv'.format(args.forget_paradigm, args.forget_paradigm, args.data_name, args.alpha, len(args.forget_class_idx), args.paradigm), index=False)
    elif args.cut_sample < 1.0:
        df.to_csv(
            './results/{}/MIA_{}_data_{}_distri_{}_fnum_{}_algo_{}_partdata_{}.csv'.format(args.forget_paradigm, args.forget_paradigm, args.data_name, args.alpha, len(args.forget_class_idx), args.paradigm, args.cut_sample), index=False)
    return


def train_shadow_model(args, case, model, proxy_client_loaders_bk, proxy_client_loaders, proxy_test_loaders):
    attack_x_train, attack_y_train, classes_train = [], [], []
    attack_x_train_user, attack_y_train_user = {}, {}
    for i in range(args.num_user):
        attack_x_train_user[i] = []
        attack_y_train_user[i] = []
    
    for i in range(args.n_shadow):
        if args.paradigm == 'fused':
            if args.forget_paradigm == 'class':
                shadow_unlearning_model = case.forget_class(copy.deepcopy(model), proxy_client_loaders, proxy_test_loaders)
            elif args.forget_paradigm == 'client':
                shadow_unlearning_model = case.forget_client_train(copy.deepcopy(model), proxy_client_loaders_bk, proxy_test_loaders)  
            elif args.forget_paradigm == 'sample':
                shadow_unlearning_model = case.forget_sample(copy.deepcopy(model), proxy_client_loaders, proxy_test_loaders)         
        elif args.paradigm == 'retrain':
            if args.forget_paradigm == 'class':
                shadow_unlearning_model = case.FL_Retrain(copy.deepcopy(model), proxy_client_loaders, proxy_test_loaders, args)
            elif args.forget_paradigm == 'client':
                shadow_unlearning_model = case.FL_Retrain(copy.deepcopy(model), proxy_client_loaders_bk, proxy_test_loaders, args)
            elif args.forget_paradigm == 'sample':
                shadow_unlearning_model = case.FL_Retrain(copy.deepcopy(model), proxy_client_loaders, proxy_test_loaders, args)
        elif args.paradigm == 'federaser':
            _, shadow_unlearning_model, _, _ = case.federated_learning_unlearning(copy.deepcopy(model), proxy_client_loaders, proxy_test_loaders, args)
        elif args.paradigm == 'exactfun':
            if args.forget_paradigm != 'client':
                shadow_unlearning_model = case.federated_unlearning(copy.deepcopy(model), proxy_client_loaders_bk, proxy_test_loaders)
            elif args.forget_paradigm == 'client':
                shadow_unlearning_model = case.federated_unlearning(copy.deepcopy(model), proxy_client_loaders, proxy_test_loaders)
        elif args.paradigm == 'eraseclient':
            shadow_unlearning_model = case.fl_unlearning(copy.deepcopy(model), proxy_client_loaders, proxy_test_loaders)

        for user in range(args.num_user):
            attack_loader = proxy_client_loaders_bk[user]
            
            for batch_idx, data in enumerate(attack_loader):
                shadow_unlearning_model.to(args.device)
                images, labels = data[0].to(args.device), data[1].to(args.device)
                outputs = shadow_unlearning_model(images)
                attack_x_train.extend(outputs.cpu().detach())
                attack_x_train_user[user].extend(outputs.cpu().detach())
                labels = labels.cpu() 
                if args.forget_paradigm == 'class':
                    matrix = np.ones(len(labels))
                    unlearn_idx = np.where(np.isin(labels, args.forget_class_idx))[0]
                    for idx in unlearn_idx:
                        matrix[idx] = 0
                    attack_y_train.extend(matrix)
                    attack_y_train_user[user].extend(matrix)
                elif args.forget_paradigm == 'client':
                    if user in args.forget_client_idx:
                        attack_y_train.extend(np.zeros(len(labels)))
                        attack_y_train_user[user].extend(np.zeros(len(labels)))
                    else:
                        attack_y_train.extend(np.ones(len(labels)))
                        attack_y_train_user[user].extend(np.ones(len(labels)))
                elif args.forget_paradigm == 'sample':
                    if (user in args.forget_client_idx) and (batch_idx <= 1):
                        attack_y_train.extend(np.zeros(len(labels)))
                        attack_y_train_user[user].extend(np.zeros(len(labels)))
                    else:
                        attack_y_train.extend(np.ones(len(labels)))
                        attack_y_train_user[user].extend(np.ones(len(labels)))
                classes_train.extend(labels)
            if args.forget_paradigm == 'client':
                attack_test_loader = proxy_test_loaders[user]
                for data in attack_test_loader:
                    images, labels = data[0].to(args.device), data[1].to(args.device)
                    labels = labels.cpu()
                    outputs = shadow_unlearning_model(images)
                    attack_x_train.extend(outputs.cpu().detach())
                    attack_x_train_user[user].extend(outputs.cpu().detach())
                    if user in args.forget_client_idx:
                        attack_y_train.extend(np.zeros(len(labels)))
                        attack_y_train_user[user].extend(np.zeros(len(labels)))
                    else:
                        attack_y_train.extend(np.ones(len(labels)))
                        attack_y_train_user[user].extend(np.ones(len(labels)))
                    classes_train.extend(labels)
            
        
        if args.forget_paradigm == 'class':
            attack_test_loader = proxy_test_loaders[0]
            for data in attack_test_loader:
                images, labels = data[0].to(args.device), data[1].to(args.device)
                outputs = shadow_unlearning_model(images)
                attack_x_train.extend(outputs.cpu().detach())
                labels = labels.cpu()

                matrix = np.ones(len(labels))
                unlearn_idx = np.where(np.isin(labels, args.forget_class_idx))[0]
                for idx in unlearn_idx:
                    matrix[idx] = 0
                attack_y_train.extend(matrix)
                classes_train.extend(labels)
        elif args.forget_paradigm == 'sample':
            attack_test_loader = proxy_test_loaders[0]
            for batch_idx, data in enumerate(attack_test_loader):
                images, labels = data[0].to(args.device), data[1].to(args.device)
                outputs = shadow_unlearning_model(images)
                attack_x_train.extend(outputs.cpu().detach())
                labels = labels.cpu()
                attack_y_train.extend(np.ones(len(labels)))
                classes_train.extend(labels)
                

    attack_x_train = np.array(attack_x_train)
    attack_y_train = np.array(attack_y_train)
    classes_train = np.array(classes_train)
    attack_x_train = attack_x_train.astype('float32')
    attack_y_train = attack_y_train.astype('int32')
    classes_train = classes_train.astype( 'int32')
    for k, v in attack_x_train_user.items():
        attack_x_train_user[k] = np.array(v)
        attack_y_train_user[k] = np.array(attack_y_train_user[k])
        attack_x_train_user[k] = attack_x_train_user[k].astype('float32')
        attack_y_train_user[k] = attack_y_train_user[k].astype('int32')

    return attack_x_train, attack_y_train, classes_train, attack_x_train_user, attack_y_train_user

def select_part_sample(args, client_all_loaders, selected_clients):
    select_client_loaders = []
    for idx in selected_clients:
        client_loader = client_all_loaders[idx]
        all_data = []

        for batch in client_loader:
            inputs, labels = batch
            all_data.append((inputs, labels))

        all_inputs = torch.cat([data[0] for data in all_data])
        all_labels = torch.cat([data[1] for data in all_data])

        unique_classes = torch.unique(all_labels)
        sampled_inputs = []
        sampled_labels = []

        for cls in unique_classes:
            class_indices = (all_labels == cls).nonzero(as_tuple=True)[0]
            total_class_samples = len(class_indices)
            sample_size = int(total_class_samples * args.cut_sample)

            if sample_size > 0:
                sampled_class_indices = random.sample(class_indices.tolist(), sample_size)
                sampled_inputs.append(all_inputs[sampled_class_indices])
                sampled_labels.append(all_labels[sampled_class_indices])

        sampled_inputs = torch.cat(sampled_inputs)
        sampled_labels = torch.cat(sampled_labels)

        dataloader = DataLoader(TensorDataset(sampled_inputs, sampled_labels), batch_size=args.test_batch_size, shuffle=True)
        select_client_loaders.append(dataloader)

    return select_client_loaders


