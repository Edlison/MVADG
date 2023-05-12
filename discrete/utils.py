# @Author  : Edlison
# @Date    : 5/11/21 19:40
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
import os
import random

cuda = True if torch.cuda.is_available() else False


def data2onehot(x):
    encoder = OneHotEncoder()
    encoder.fit(x)
    res = encoder.transform(x).toarray()
    return res


def softmax2argmax(x):
    res = torch.argmax(x, dim=1)
    return res


def features2argmax(x):
    features_splited = torch.split(x, 2, dim=1)
    features = []
    for feature in features_splited:
        features.append(torch.argmax(feature, dim=1).reshape([-1, 1]))
    res = torch.cat(features, dim=1)
    return res


def sample_z(shape=64, latent_dim=10, n_c=10, fix_class=-1, req_grad=False):  # shape就是img[0], 也就是batch_size
    assert (fix_class == -1 or (fix_class >= 0 and fix_class < n_c)), "Requested class %i outside bounds." % fix_class
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    zn = Tensor(0.75 * np.random.normal(0, 1, (shape, latent_dim)))
    zc_FT = Tensor(shape, n_c).fill_(0)  # FloatTensor类型的张量，是empty的特例。初始化形状，用0填充。torch.zeros
    zc_idx = torch.empty(shape, dtype=torch.long)
    if (fix_class == -1):
        zc_idx = zc_idx.random_(n_c).cuda() if cuda else zc_idx.random_(n_c)  # 用一个离散均匀分布来填充当前的张量。[from 0, to n_c]
        zc_FT = zc_FT.scatter_(1, zc_idx.unsqueeze(1), 1.)
    else:
        zc_idx[:] = fix_class
        zc_FT[:, fix_class] = 1

        zc_idx = zc_idx.cuda() if cuda else zc_idx
        zc_FT = zc_FT.cuda() if cuda else zc_idx
    zc = zc_FT
    return zn, zc, zc_idx


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def save_models(*o):
    dir = './out/models'
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)
    for model in o:
        torch.save(model.state_dict(), os.path.join(dir, model.name + '.pt'))
    print('models saved in {}'.format(dir))


class ZooDataset(Dataset):
    f = '../../Datasets/processed/zoo.pt'
    f_split = '../../Datasets/processed/zoo_split.pt'
    cls_attr_ca_anom = '../../Datasets/processed/zoo_cls_attr_ca_anom.pt'

    def __init__(self, anom=False):
        self.anom = anom
        if not anom:
            self.data, self.labels = torch.load(self.f)
            print('load ', self.f)
            self.data = data2onehot(self.data)
        else:
            self.data1, self.data2, self.labels = torch.load(self.cls_attr_ca_anom)
            print('load ', self.cls_attr_ca_anom)
            self.data1 = data2onehot(self.data1)
            self.data2 = data2onehot(self.data2)

    def __getitem__(self, index):
        if not self.anom:
            data, labels = self.data[index], self.labels[index]
            return data, labels
        else:
            data1, data2, labels = self.data1[index], self.data2[index], self.labels[index]
            return data1, data2, labels

    def __len__(self):
        return len(self.labels)

    def gen_split(self):
        data, labels = torch.load(self.f)
        data1, data2 = torch.chunk(data, 2, dim=1)
        file = (data1, data2, labels)
        with open(self.f_split, 'wb') as f:
            torch.save(file, f)
            print('zoo split success in {}'.format(self.f_split))

    def gen_cls_attr_ca_anom(self):
        """
        类异常 + 属性异常 + 类属性
        """
        data1, data2, labels = torch.load(self.f_split)
        cls_anom = [0, 1, -1, -2, 50]
        attr_anom = [2, 3, -3, -4, 51]
        ca_anom = [4, 5, -5, -6, 52]
        torch.zero_(labels).type(dtype=torch.int)
        # cls anom
        cls_1 = data1[cls_anom[0]]
        for i in range(len(cls_anom) - 1):
            data1[cls_anom[i]] = data1[cls_anom[i + 1]]
        data1[cls_anom[-1]] = cls_1
        labels[cls_anom] = 1
        # attr anom
        for i in attr_anom:
            data1[i] = torch.randint(low=0, high=2, size=data1[i].shape)
        labels[attr_anom] = 1
        # ca anom
        ca_1 = data1[ca_anom[0]]
        for i in range(len(ca_anom) - 1):
            data1[ca_anom[i]] = data1[ca_anom[i + 1]]
        data1[ca_anom[-1]] = ca_1
        for i in ca_anom:
            data2[i] = torch.randint(low=0, high=2, size=data2[i].shape)
        labels[ca_anom] = 1
        # save
        file = (data1, data2, labels)
        with open(self.cls_attr_ca_anom, 'wb') as f:
            torch.save(file, f)
            print('3types anom success in {}'.format(self.cls_attr_ca_anom))


class WineDataset(Dataset):
    ori_f = '../../Datasets/wine/wine.data'
    tar_f = '../../Datasets/processed/wine.pt'
    split_f = '../../Datasets/processed/wine_split_anom.pt'
    cls_anom = '../../Datasets/processed/wine_cls_anom.pt'
    attr_anom = '../../Datasets/processed/wine_attr_anom.pt'
    cls_attr_ca_anom = '../../Datasets/processed/wine_cls_attr_ca_anom.pt'

    def __init__(self, anom=False, gen=False):
        self.anom = anom
        if gen:
            self.gen_wine()
        else:
            if not anom:
                self.data, self.labels = torch.load(self.tar_f)
                print('load ', self.tar_f)
                self.data = MinMaxScaler(feature_range=(-1, 1)).fit_transform(self.data)
            else:
                self.data1, self.data2, self.labels = torch.load(self.cls_attr_ca_anom)
                print('load ', self.cls_attr_ca_anom)
                self.data1 = MinMaxScaler(feature_range=(-1, 1)).fit_transform(self.data1)
                self.data2 = MinMaxScaler(feature_range=(-1, 1)).fit_transform(self.data2)

    def __getitem__(self, index):
        if not self.anom:
            data, label = self.data[index], self.labels[index]
            return data, label
        else:
            data1, data2, label = self.data1[index], self.data2[index], self.labels[index]
            return data1, data2, label

    def __len__(self):
        return len(self.labels)

    def gen_wine(self):
        data = []
        labels = []
        with open(self.ori_f, 'r') as f:
            for line in f:
                line = line.strip('\n')
                line = line.split(',')
                data.append([float(i) for i in line[1:]])
                labels.append(int(line[0]))
        data_temp = []
        data_temp.extend(data[:45])
        data_temp.extend(data[60:105])
        data_temp.extend(data[131:176])
        labels_temp = []
        labels_temp.extend(labels[:45])
        labels_temp.extend(labels[60:105])
        labels_temp.extend(labels[131:176])

        data = torch.tensor(data_temp)
        labels = torch.tensor(labels_temp)
        file = (data, labels)
        torch.save(file, self.tar_f)
        print('save successed in {}.'.format(self.tar_f))

    def split_wine(self, anom_num_per_feature=5):
        data, labels = torch.load(self.tar_f)
        data1, data2 = torch.chunk(data, 2, dim=1)
        feature_num = 45
        anom_index = random.sample(range(feature_num), anom_num_per_feature)
        torch.zero_(labels).type(dtype=torch.int)
        for i in anom_index:
            temp = torch.clone(data1[i])
            data1[i] = data1[i + feature_num]
            data1[i + feature_num] = data1[i + feature_num * 2]
            data1[i + feature_num * 2] = temp
            labels[i], labels[i + feature_num], labels[i + feature_num * 2] = 1, 1, 1
        file = (data1, data2, labels)
        with open(self.split_f, 'wb') as f:
            torch.save(file, f)
            print('split anom success in {}'.format(self.split_f))

    def gen_cls_attr_ca_anom(self):
        """
        类异常 + 属性异常 + 类属性
        """
        data, labels = torch.load(self.tar_f)
        data1, data2 = torch.chunk(data, 2, dim=1)
        feature_num = 45
        anom_index = random.sample(range(feature_num), 9)
        cls_anom = anom_index[:3]
        attr_anom = anom_index[3:6]
        ca_anom = anom_index[6:]
        torch.zero_(labels).type(dtype=torch.int)
        for i in cls_anom:
            temp = torch.clone(data1[i])
            data1[i] = data1[i + feature_num]
            data1[i + feature_num] = data1[i + feature_num * 2]
            data1[i + feature_num * 2] = temp
            labels[i], labels[i + feature_num], labels[i + feature_num * 2] = 1, 1, 1
        for i in attr_anom:
            data1[i] = torch.randn(data1[i].shape)
            data1[i + feature_num] = torch.randn(data1[i].shape)
            data1[i + feature_num * 2] = torch.randn(data1[i].shape)
            labels[i], labels[i + feature_num], labels[i + feature_num * 2] = 1, 1, 1
        for i in ca_anom:
            temp = torch.clone(data1[i])
            data1[i] = data1[i + feature_num]
            data1[i + feature_num] = data1[i + feature_num * 2]
            data1[i + feature_num * 2] = temp
            data2[i] = torch.randn(data2[i].shape)
            data2[i + feature_num] = torch.randn(data2[i].shape)
            data2[i + feature_num * 2] = torch.randn(data2[i].shape)
            labels[i], labels[i + feature_num], labels[i + feature_num * 2] = 1, 1, 1
        file = (data1, data2, labels)
        with open(self.cls_attr_ca_anom, 'wb') as f:
            torch.save(file, f)
            print('3types anom success in {}'.format(self.cls_attr_ca_anom))
