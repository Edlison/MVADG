import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

DATASET_FN_DICT = {'mnist': datasets.MNIST,
                   'fashion-mnist': datasets.FashionMNIST
                   }

dataset_list = DATASET_FN_DICT.keys()


def get_dataset(dataset_name='mnist'):
    if dataset_name in DATASET_FN_DICT:
        fn = DATASET_FN_DICT[dataset_name]
        return fn
    else:
        raise ValueError('Invalid dataset, {}, entered. Must be '
                         'in {}'.format(dataset_name, DATASET_FN_DICT.keys()))


def get_dataloader(dataset_name='mnist', data_dir='./datasets/mnist', batch_size=64, train_set=True, num_workers=1):
    dset = get_dataset(dataset_name)

    dataloader = torch.utils.data.DataLoader(
        dset(data_dir, train=train_set, download=False, transform=transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
             ),
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader


class MNISTDataset(Dataset):
    tar_f = '../data/MNIST/processed/training.pt'

    def __init__(self):
        self.imgs, self.labels = torch.load(self.tar_f)

    def __getitem__(self, idx):
        image, label = self.imgs[idx], self.labels[idx]
        return image, label

    def __len__(self):
        return len(self.labels)


class MNISTAnomDataset(Dataset):
    shape = [28, 28]
    ori_f = '../data/MNIST/processed/training.pt'
    tar_dir = '/home/rain/Projects/MVADG/Datasets/processed/'

    def __init__(self, f, pixel=16, transform=None):
        file = self.tar_dir + f
        self.imgs, self.labels = torch.load(file)
        self.pixel = pixel
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.imgs[idx], self.labels[idx]
        if self.pixel == 28:
            pass
        elif self.pixel == 16:
            image = self._down(image)
        else:
            raise ValueError('pixel err.')

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)

    def gen_nc(self, *digits, num=500):
        data, labels = torch.load(self.ori_f)
        res_data = torch.empty(len(digits) * num, 1, 28, 28)
        res_labels = torch.empty(len(digits) * num)
        index = 0
        j = 0
        for digit in digits:
            each_num = num
            for i, label in enumerate(labels):
                if digit == label and each_num > 0:
                    temp = Image.fromarray(data[i].numpy(), mode='L')
                    res_data[index] = transforms.ToTensor()(temp)
                    res_labels[index] = j
                    each_num -= 1
                    index += 1
            j += 1
        tar_f = self.tar_dir + 'mnist_{}_{}.pt'.format(len(digits), num)
        file = (res_data, res_labels)
        with open(tar_f, 'wb') as f:
            torch.save(file, f)
            print('gen {}cls file success.'.format(len(digits)))

    def _down(self, img):
        """
        28 * 28 -> 16 * 16
        """
        img = Image.fromarray(img.squeeze().numpy())
        img = img.resize((16, 16))
        img_arr = np.array(img)
        img_arr = torch.Tensor(img_arr)
        img_arr.unsqueeze_(dim=0)
        return img_arr


class UspsDataset(Dataset):
    ori_f = '../Datasets/usps/usps'
    tar_f = '../Datasets/processed/usps.pt'

    def __init__(self):
        if not os.path.exists(self.tar_f):
            raise FileExistsError('usps processed file not exists.')
        self.data, self.labels = torch.load(self.tar_f)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        return data, label

    def __len__(self):
        return len(self.labels)

    def gen_usps(self):
        with open(self.ori_f, 'r') as f:
            num = len(f.readlines())
            data = torch.empty(num, 16, 16, dtype=torch.float32)
            labels = torch.empty(num, dtype=torch.int)
        with open(self.ori_f, 'r') as f:
            index = 0
            for line in f:
                line.strip()
                line_list = line.split(' ')
                temp = torch.tensor([float(i[i.find(':') + 1:]) for i in line_list[1:257]])
                data[index] = temp.reshape(16, 16)
                labels[index] = int(line_list[0]) - 1
                index += 1
        print(data.shape)
        print(labels.shape)
        file = (data, labels)
        with open(self.tar_f, 'wb') as f:
            torch.save(file, f)
            print('save processed success.')


class UspsAnomDataset(Dataset):
    ori_f = '../Datasets/processed/usps.pt'
    tar_dir = '/home/rain/Projects/MVADG/Datasets/processed/'

    def __init__(self, f):
        file = self.tar_dir + f
        self.data, self.labels = torch.load(file)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def gen_nc(self, *digits, num=500):
        data, labels = torch.load(self.ori_f)
        res_data = torch.empty(len(digits) * num, 1, 16, 16)
        res_labels = torch.empty(len(digits) * num)
        index = 0
        j = 0
        for digit in digits:
            each_num = num
            for i, label in enumerate(labels):
                if digit == label and each_num > 0:
                    temp = Image.fromarray(data[i].numpy())
                    res_data[index] = transforms.ToTensor()(temp)
                    res_labels[index] = j
                    each_num -= 1
                    index += 1
            j += 1
        tar_f = self.tar_dir + 'usps_{}_{}.pt'.format(len(digits), num)
        file = (res_data, res_labels)
        print(res_data.shape)
        with open(tar_f, 'wb') as f:
            torch.save(file, f)
            print('gen {}cls file success.'.format(len(digits)))


class MultiDataset(Dataset):
    dataset = {'mnist': MNISTAnomDataset, 'usps': UspsAnomDataset}
    tar_dir = '../Datasets/processed/'

    def __init__(self, ds1='mnist', f1='mnist_3_500.pt', ds2='usps', f2='usps_3_500.pt'):
        self.dataset1 = self.dataset[ds1](f1)
        self.dataset2 = self.dataset[ds2](f2)

    def __getitem__(self, idx):
        return self.dataset1[idx], self.dataset2[idx]

    def __len__(self):
        return len(self.dataset1.labels)


def get_loader(ds='usps', batch=64, shuffle=False):
    dataset = {'usps': UspsDataset(), 'mnist': MNISTDataset()}
    if ds in dataset:
        return DataLoader(dataset[ds], batch_size=batch, shuffle=shuffle)
    else:
        raise ValueError('no dataset.')


def get_anom_loader(ds='mnist', processed='anom.pt', batch_num=64, shuffle=False):
    dataset = {'mnist': MNISTAnomDataset, 'usps': UspsAnomDataset}
    if ds in dataset:
        return DataLoader(dataset[ds](processed), batch_size=batch_num, shuffle=shuffle)
    else:
        raise ValueError('no dataset.')


def get_multi_loader(batch=64, shuffle=False):
    return DataLoader(MultiDataset(), batch_size=batch, shuffle=shuffle)


def get_multi_anom_loader(batch=64, shuffle=False):
    return DataLoader(MultiDataset(ds1='mnist', f1='mnist_3_500_anom_cls.pt', ds2='usps', f2='usps_3_500.pt'),
                      batch_size=batch, shuffle=shuffle)


def get_usps_loader(batch_size=256, shuffle=True):
    return DataLoader(datasets.USPS('../data/USPS', transform=transforms.Compose([
        transforms.Resize([16, 16]),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])), batch_size=batch_size, shuffle=shuffle)


def gen_anom_cls(processed='', norm_num=500, anom_num=15):
    ori_f = '../Datasets/processed/' + processed
    tar_f = '../Datasets/processed/' + processed[:-3] + '_anom_cls.pt'
    imgs, labels = torch.load(ori_f)
    par = int(anom_num / 3)
    anom_idx = random.sample(range(norm_num), par)
    torch.zero_(labels).type(dtype=torch.int)
    for i in anom_idx:
        temp = torch.clone(imgs[i])
        imgs[i] = imgs[i + norm_num]
        imgs[i + norm_num] = imgs[i + norm_num * 2]
        imgs[i + norm_num * 2] = temp
        labels[i], labels[i + norm_num], labels[i + norm_num * 2] = 1, 1, 1
    file = (imgs, labels)
    with open(tar_f, 'wb') as f:
        torch.save(file, f)
        print('gen anom successed in {}.'.format(tar_f))


def gen_anom_attr(processed='', anom_num=15):
    ori_f = '../Datasets/processed/' + processed
    tar_f = '../Datasets/processed/' + processed[:-3] + '_anom_attr.pt'
    imgs, labels = torch.load(ori_f)
    shape = [1, 16, 16]
    anom_idx = random.sample(range(len(labels)), anom_num)
    torch.zero_(labels).type(dtype=torch.int)
    for i in range(len(labels)):
        if i in anom_idx:
            imgs[i] = torch.rand(shape)
            labels[i] = 1
    file = (imgs, labels)
    with open(tar_f, 'wb') as f:
        torch.save(file, f)
        print('gen anom successed in {}.'.format(tar_f))
