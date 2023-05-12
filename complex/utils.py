# @Author  : Edlison
# @Date    : 6/7/21 20:19
import os
import sys
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy.io as sio


def show(data, row, col):
    import matplotlib.pyplot as plt
    len1 = data.shape[0]
    len2 = row * col
    for i in range(min(len1, len2)):
        plt.subplot(row, col, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(data[i].squeeze())
    plt.show()


def dataloader_v2(dataset='mnist', anom='005', feat1='lbp', feat2='glcm', batch=64, shuffle=False):
    if dataset == 'mnist':
        return DataLoader(MNISTDataset(anom=anom, feat1=feat1, feat2=feat2), batch_size=batch, shuffle=shuffle)
    elif dataset == 'fashionmnist':
        return DataLoader(FashionMNISTDataset(anom=anom, feat1=feat1, feat2=feat2), batch_size=batch, shuffle=shuffle)
    else:
        raise ValueError('no dataset')


def dataloader_v3(dataset='mnist', anom='005', feat1='ori', feat2='lbp', feat3='glcm', batch=64, shuffle=False):
    if dataset == 'mnist':
        return DataLoader(MNISTDataset(anom=anom, feat1=feat1, feat2=feat2, feat3=feat3), batch_size=batch,
                          shuffle=shuffle)
    elif dataset == 'fashionmnist':
        return DataLoader(FashionMNISTDataset(anom=anom, feat1=feat1, feat2=feat2, feat3=feat3), batch_size=batch,
                          shuffle=shuffle)
    else:
        raise ValueError('no dataset')


def save_net(net, dir):
    tar_f = os.path.join(dir, 'net.pt')
    torch.save(net.state_dict(), tar_f)
    print('net saved in {}'.format(tar_f))


def draw_loss(ge_loss1, ge_loss2, d_loss1, d_loss2, dir):
    tar_f = os.path.join(dir, 'loss.png')
    epoch = range(0, len(ge_loss1))
    plt.figure()
    plt.title('loss')
    plt.plot(epoch, ge_loss1, color='blue', label='ge_loss1')
    plt.plot(epoch, ge_loss2, color='skyblue', label='ge_loss2')
    plt.plot(epoch, d_loss1, color='green', label='d_loss1')
    plt.plot(epoch, d_loss2, color='orange', label='d_loss2')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(tar_f)
    plt.show()


def draw_loss3v(ge_loss1, ge_loss2, ge_loss3, d_loss1, d_loss2, d_loss3, dir):
    tar_f = os.path.join(dir, 'loss.png')
    epoch = range(0, len(ge_loss1))
    plt.figure()
    plt.title('loss')
    plt.plot(epoch, ge_loss1, color='blue', label='ge_loss1')
    plt.plot(epoch, ge_loss2, color='skyblue', label='ge_loss2')
    plt.plot(epoch, ge_loss3, color='purple', label='ge_loss2')
    plt.plot(epoch, d_loss1, color='green', label='d_loss1')
    plt.plot(epoch, d_loss2, color='orange', label='d_loss2')
    plt.plot(epoch, d_loss3, color='yellow', label='d_loss2')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(tar_f)
    plt.show()


def draw_auc(auc, dir):
    tar_f = os.path.join(dir, 'auc.png')
    epoch = range(0, len(auc))
    plt.figure()
    plt.title('auc')
    plt.plot(epoch, auc, color='red', label='auc')
    for x, y in zip(epoch, auc):
        plt.text(x, y, y, ha='center', fontsize=10)
    plt.xlabel('epochs')
    plt.ylabel('auc')
    plt.legend()
    plt.savefig(tar_f)
    plt.show()
    if len(auc) > 0:
        auc.sort(reverse=True)
        print('auc best: ', auc[0])


class MNISTDataset(Dataset):
    feat1v_f = '../../data/mnist_custom/processed/mnist_{}.pt'
    feat2v_f = '../../data/mnist_custom/processed/mnist_{}_{}.pt'
    feat3v_f = '../../data/mnist_custom/processed/mnist_{}_{}_{}.pt'
    anom_feat2v_f = '../../data/mnist_custom/processed/mnist_{}_{}_anom{}.pt'
    anom_feat3v_f = '../../data/mnist_custom/processed/mnist_{}_{}_{}_anom{}.pt'
    feat_list = {'ori', 'lbp', 'glcm'}
    anom_list = {'015', '005', '001', '005_0', '005_20', '005_40', '005_60', '005_80', '005_100'}

    def __init__(self, rt=True, anom='', feat1='ori', feat2='lbp', feat3=''):
        if rt:
            return
        if feat3 == '':
            self.view = 2
            if anom and anom in self.anom_list:
                self.data1, self.data2, self.labels = torch.load(self.anom_feat2v_f.format(feat1, feat2, anom))
            else:
                self.data1, self.data2, self.labels = torch.load(self.feat2v_f.format(feat1, feat2))
        else:
            self.view = 3
            if anom and anom in self.anom_list:
                self.data1, self.data2, self.data3, self.labels = torch.load(
                    self.anom_feat3v_f.format(feat1, feat2, feat3, anom))
            else:
                self.data1, self.data2, self.data3, self.labels = torch.load(self.feat3v_f.format(feat1, feat2, feat3))

    def __getitem__(self, index):
        if self.view == 2:
            data1, data2, label = self.data1[index], self.data2[index], self.labels[index]
            return data1, data2, label
        else:
            data1, data2, data3, label = self.data1[index], self.data2[index], self.data3[index], self.labels[index]
            return data1, data2, data3, label

    def __len__(self):
        return len(self.labels)

    def gen(self, feat1='lbp', feat2='glcm', feat3=''):
        """
        根据需要选择生成2view合集 lbp_glcm
        """
        if feat3 == '':
            data1, _ = torch.load(self.feat1v_f.format(feat1))
            data2, _ = torch.load(self.feat1v_f.format(feat2))
            scaler = MinMaxScaler()
            trans = transforms.Compose([
                transforms.ToTensor()
            ])
            res_data1 = torch.empty(data1.shape)
            res_data2 = torch.empty(data2.shape)
            for i in range(len(data1)):
                res_data1[i] = trans(scaler.fit_transform(data1[i]))
                res_data2[i] = trans(scaler.fit_transform(data2[i]))
            res_data1.unsqueeze_(dim=1)
            res_data2.unsqueeze_(dim=1)
            file = (res_data1, res_data2, _)
            with open(self.feat2v_f.format(feat1, feat2), 'wb') as f:
                torch.save(file, f)
            print('gen {} success.'.format(self.feat2v_f.format(feat1, feat2)))
        else:
            data1, _ = torch.load(self.feat1v_f.format(feat1))
            data2, _ = torch.load(self.feat1v_f.format(feat2))
            data3, _ = torch.load(self.feat1v_f.format(feat3))
            scaler = MinMaxScaler()
            trans = transforms.Compose([
                transforms.ToTensor()
            ])
            res_data1 = torch.empty(data1.shape)
            res_data2 = torch.empty(data2.shape)
            res_data3 = torch.empty(data3.shape)
            for i in range(len(data1)):
                res_data1[i] = trans(scaler.fit_transform(data1[i]))
                res_data2[i] = trans(scaler.fit_transform(data2[i]))
                res_data3[i] = trans(scaler.fit_transform(data3[i]))
            res_data1.unsqueeze_(dim=1)
            res_data2.unsqueeze_(dim=1)
            res_data3.unsqueeze_(dim=1)
            file = (res_data1, res_data2, res_data3, _)
            with open(self.feat3v_f.format(feat1, feat2, feat3), 'wb') as f:
                torch.save(file, f)
            print('gen {} success.'.format(self.feat3v_f.format(feat1, feat2, feat3)))

    def gen_cls_attr_ca_anom(self, anom='', feat1='lbp', feat2='glcm', feat3=''):
        """
        类异常 + 属性异常 + 类属性
        """
        feature_num = 60
        if feat3 == '':
            data1, data2, labels = torch.load(self.feat2v_f.format(feat1, feat2))
        else:
            data1, data2, data3, labels = torch.load(self.feat3v_f.format(feat1, feat2, feat3))
        if anom in self.anom_list:
            if '015' == anom:
                anom_num = feature_num * 0.15
            elif '005' == anom:
                anom_num = feature_num * 0.05
            elif '001' == anom:
                anom_num = feature_num * 0.01
            else:
                raise ValueError('ratio not exist.')
            anom_index = random.sample(range(feature_num), int(anom_num))  # 015 - 9000/3, 005 - 3000/3, 001 - 600/3
            cls_anom, attr_anom, ca_anom = np.array_split(np.array(anom_index), 3)
            print('anom index ', anom_index)
            print('anom num ', len(anom_index))
        else:
            raise ValueError('anom not exist.')
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

        if feat3 == '':
            file = (data1, data2, labels)
            with open(self.anom_feat2v_f.format(feat1, feat2, anom), 'wb') as f:
                torch.save(file, f)
                print('anom generate success in {}'.format(self.anom_feat2v_f.format(feat1, feat2, anom)))
        else:
            file = (data1, data2, data3, labels)
            with open(self.anom_feat3v_f.format(feat1, feat2, feat3, anom), 'wb') as f:
                torch.save(file, f)
                print('anom generate success in {}'.format(self.anom_feat3v_f.format(feat1, feat2, feat3, anom)))

    def gen_cls_attr(self, anom='', feat1='lbp', feat2='glcm', feat3=''):
        """
        类异常 + 属性异常（按不同比例）
        """
        feature_num = 20000  # 200 -> 60; 60000 -> 20000
        if feat3 == '':
            data1, data2, labels = torch.load(self.feat2v_f.format(feat1, feat2))
        else:
            data1, data2, data3, labels = torch.load(self.feat3v_f.format(feat1, feat2, feat3))
        if anom in self.anom_list:
            if '005' in anom:
                anom_num = feature_num * 0.05
                if '_0' in anom:
                    cls_num = anom_num * 0
                    attr_num = anom_num - cls_num
                elif '_20' in anom:
                    cls_num = anom_num * 0.2
                    attr_num = anom_num - cls_num
                elif '_40' in anom:
                    cls_num = anom_num * 0.4
                    attr_num = anom_num - cls_num
                elif '_60' in anom:
                    cls_num = anom_num * 0.6
                    attr_num = anom_num - cls_num
                elif '_80' in anom:
                    cls_num = anom_num * 0.8
                    attr_num = anom_num - cls_num
                elif '_100' in anom:
                    cls_num = anom_num * 1
                    attr_num = anom_num - cls_num
                else:
                    raise ValueError('ratio not exist.')
            else:
                raise ValueError('ratio not exist.')
            anom_index = random.sample(range(feature_num), int(anom_num))  # 015 - 9000/3, 005 - 3000/3, 001 - 600/3
            cls_anom = anom_index[:int(cls_num)]
            attr_anom = anom_index[int(cls_num):]
            print('anom index ', anom_index)
            print('cls anom ', cls_anom)
            print('cls num ', cls_num)
            print('cls len ', len(cls_anom))
            print('attr anom ', attr_anom)
            print('attr num ', attr_num)
            print('attr len ', len(attr_anom))
            print('anom num ', len(anom_index))
        else:
            raise ValueError('anom not exist.')
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

        if feat3 == '':
            file = (data1, data2, labels)
            with open(self.anom_feat2v_f.format(feat1, feat2, anom), 'wb') as f:
                torch.save(file, f, _use_new_zipfile_serialization=False)
                print('anom generate success in {}'.format(self.anom_feat2v_f.format(feat1, feat2, anom)))
        else:
            file = (data1, data2, data3, labels)
            with open(self.anom_feat3v_f.format(feat1, feat2, feat3, anom), 'wb') as f:
                torch.save(file, f, _use_new_zipfile_serialization=False)
                print('anom generate success in {}'.format(self.anom_feat3v_f.format(feat1, feat2, feat3, anom)))


class FashionMNISTDataset(MNISTDataset):
    feat1v_f = '../../data/FashionMNIST/processed/fashionmnist_{}.pt'
    feat2v_f = '../../data/FashionMNIST/processed/fashionmnist_{}_{}.pt'
    feat3v_f = '../../data/FashionMNIST/processed/fashionmnist_{}_{}_{}.pt'
    anom_feat2v_f = '../../data/FashionMNIST/processed/fashionmnist_{}_{}_anom{}.pt'
    anom_feat3v_f = '../../data/FashionMNIST/processed/fashionmnist_{}_{}_{}_anom{}.pt'

    def __init__(self, rt=True, anom='', feat1='ori', feat2='lbp', feat3=''):
        super().__init__(rt, anom, feat1, feat2, feat3)

    def get(self):
        file = '../../data/FashionMNIST/processed/fashionmnist_ori_glcm_anom015.pt'
        data1, data2, label = torch.load(file)
        print(data1.shape)
        print(data2.shape)
        print(label.shape)
        print(label[:25])
        from v6.Eval import show
        show(data1[:25], 5, 5)
        show(data2[:25], 5, 5)


def _normalize():
    global ori_f, lbp_f, glcm_f
    ori_f = '../../data/MNIST/processed/training.pt'
    lbp_f = '../../data/mnist_custom/processed/mnist_lbp.pt'
    glcm_f = '../../data/mnist_custom/processed/mnist_glcm.pt'
    ori, _ = torch.load(ori_f)
    lbp, _ = torch.load(lbp_f)
    glcm, _ = torch.load(glcm_f)
    trans = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    from v6.Eval import show
    # show(ori[:25], 5, 5)
    # show(lbp[:25], 5, 5)
    # show(glcm[:25], 5, 5)
    print('before')
    print('ori before ', ori[0])
    print('lbp before ', lbp[0])
    print('glcm before ', glcm[0])
    res_ori = torch.empty(ori.shape)
    res_lbp = torch.empty(lbp.shape)
    res_glcm = torch.empty(glcm.shape)
    for i in range(len(lbp)):
        scaler = MinMaxScaler()
        res_ori[i] = trans(scaler.fit_transform(ori[i]))
        res_lbp[i] = trans(scaler.fit_transform(lbp[i]))
        res_glcm[i] = trans(scaler.fit_transform(glcm[i]))
    print('after')
    print('ori after ', res_ori[0])
    print('lbp after ', res_lbp[0])
    print('glcm after ', res_glcm[0])


def _anomaly():
    loader = dataloader_v3(dataset='fashionmnist', anom='005_100', feat1='ori', feat2='lbp', feat3='glcm', batch=100,
                           shuffle=False)
    from v6.Eval import show
    data1, data2, data3, labels = next(iter(loader))

    print('labels ', labels)
    show(data1, 10, 10)
    show(data2, 10, 10)
    show(data3, 10, 10)


def tensor2mat(dataset='mnist', anom='015', feat1='ori', feat2='lbp', feat3=''):
    # file_FashionMNIST = '../../data/FashionMNIST/processed/fashionmnist_{}_{}_anom{}_min.pt'.format(feat1, feat2, anom)
    # file_MNIST = '../../data/mnist_custom/processed/mnist_{}_{}_anom{}_min.pt'.format(feat1, feat2, anom)
    file_FashionMNIST = '../../data/FashionMNIST/processed/fashionmnist_{}_{}_{}_anom{}_min.pt'.format(feat1, feat2,
                                                                                                       feat3, anom)
    file_MNIST = '../../data/mnist_custom/processed/mnist_{}_{}_{}_anom{}_min.pt'.format(feat1, feat2, feat3, anom)
    if dataset == 'mnist':
        file = file_MNIST
    else:
        file = file_FashionMNIST
    # data1, data2, labels = torch.load(file)
    data1, data2, data3, labels = torch.load(file)
    data1 = np.array(data1.reshape([200, -1]).transpose(0, 1))
    data2 = np.array(data2.reshape([200, -1]).transpose(0, 1))
    data3 = np.array(data3.reshape([200, -1]).transpose(0, 1))

    labels = np.array(labels.unsqueeze(1))

    # sio.savemat('../../data/mat/{}_{}_{}_{}.mat'.format(dataset, feat1, feat2, anom), {'data1': data1, 'data2': data2, 'labels': labels})
    sio.savemat('../../data/mat/{}_{}_{}_{}_{}.mat'.format(dataset, feat1, feat2, feat3, anom), {'data1': data1,
                                                                                                 'data2': data2,
                                                                                                 'data3': data3,
                                                                                                 'labels': labels})
    print('gen {} success'.format(file))


def classLabel():
    file = '../../data/MNIST/processed/training.pt'
    _, labels = torch.load(file)
    labels = np.array(labels.unsqueeze(1))
    sio.savemat('../../data/mat/classLabel.mat', {'classLabel': labels})
    print('gen classLabel success')


def ds_min():
    # tensor2mat(dataset='mnist', anom='005_100', feat1='ori', feat2='lbp')
    # classLabel()
    # feat1v_f = '../../data/mnist_custom/processed/mnist_{}_min.pt'
    # feat2v_f = '../../data/mnist_custom/processed/mnist_{}_{}_min.pt'
    # feat3v_f = '../../data/mnist_custom/processed/mnist_{}_{}_{}_min.pt'
    # anom_feat2v_f = '../../data/mnist_custom/processed/mnist_{}_{}_anom{}_min.pt'
    # anom_feat3v_f = '../../data/mnist_custom/processed/mnist_{}_{}_{}_anom{}_min.pt'
    #
    feat1v_f = '../../data/FashionMNIST/processed/fashionmnist_{}_min.pt'
    feat2v_f = '../../data/FashionMNIST/processed/fashionmnist_{}_{}_min.pt'
    feat3v_f = '../../data/FashionMNIST/processed/fashionmnist_{}_{}_{}_min.pt'
    anom_feat2v_f = '../../data/FashionMNIST/processed/fashionmnist_{}_{}_anom{}_min.pt'
    anom_feat3v_f = '../../data/FashionMNIST/processed/fashionmnist_{}_{}_{}_anom{}_min.pt'
    ds = MNISTDataset()
    ds.feat1v_f = feat1v_f
    ds.feat2v_f = feat2v_f
    ds.feat3v_f = feat3v_f
    ds.anom_feat2v_f = anom_feat2v_f
    ds.anom_feat3v_f = anom_feat2v_f
    # ds.gen_cls_attr_ca_anom(anom='015', feat1='ori', feat2='lbp', feat3='glcm')
    # ds.gen_cls_attr(anom='005_100', feat1='ori', feat2='lbp', feat3='glcm')
    f1, f2, f3, _ = torch.load(anom_feat3v_f.format('ori', 'lbp', 'glcm', '005_100'))
    print('shape ', f1.shape, f2.shape, f3.shape, _.shape)
    tensor2mat(dataset='fashionmnist', anom='005_100', feat1='ori', feat2='lbp', feat3='glcm')


def ap():
    from sklearn.cluster import affinity_propagation, AffinityPropagation

    # anom_feat2v_f = '../../data/mnist_custom/processed/mnist_{}_{}_anom{}_min.pt'
    # anom_feat2v_f = '../../data/FashionMNIST/processed/fashionmnist_{}_{}_anom{}_min.pt'
    # anom_feat3v_f = '../../data/mnist_custom/processed/mnist_{}_{}_{}_anom{}_min.pt'
    anom_feat3v_f = '../../data/FashionMNIST/processed/fashionmnist_{}_{}_{}_anom{}_min.pt'
    # f1, f2, anom_labels = torch.load(anom_feat2v_f.format('ori', 'lbp', '005_100'))
    f1, f2, f3, anom_labels = torch.load(anom_feat3v_f.format('ori', 'lbp', 'glcm', '005_100'))

    ap = AffinityPropagation(random_state=1)
    f1 = np.array(f1.reshape([200, -1]))
    f2 = np.array(f2.reshape([200, -1]))
    f3 = np.array(f3.reshape([200, -1]))
    pred_labels1 = ap.fit_predict(f1)
    pred_labels2 = ap.fit_predict(f2)
    pred_labels3 = ap.fit_predict(f3)

    print('pred labels: ', pred_labels1)
    print('pred labels: ', pred_labels2)

    from v6.discrete.test import vote_score, roc_auc_score

    # 计算cls score
    # voted_label1, voted_label2 = vote_score(pred_labels1, pred_labels2)
    # cls_score = np.zeros([len(anom_labels), ])
    # for i in range(len(voted_label1)):
    #     if voted_label1[i] != voted_label2[i]:
    #         cls_score[i] += 0.5

    voted_label1, voted_label2 = vote_score(pred_labels1, pred_labels2)
    _, voted_label3 = vote_score(voted_label2, pred_labels3)
    cls_score = np.zeros([len(anom_labels), ])
    for i in range(len(voted_label1)):
        if not voted_label1[i] == voted_label2[i] == voted_label3[i]:
            cls_score[i] += 0.5

    auc = roc_auc_score(anom_labels, cls_score)
    print('auc ', auc)


def raw_to_processed():
    from torchvision.datasets.mnist import read_image_file, read_label_file
    raw_folder = '../../data/FashionMNIST/raw'
    processed_folder = '../../data/FashionMNIST/processed'
    training_set = (
        read_image_file(os.path.join(raw_folder, 'train-images-idx3-ubyte')),
        read_label_file(os.path.join(raw_folder, 'train-labels-idx1-ubyte'))
    )
    with open(os.path.join(processed_folder, 'training.pt'), 'wb') as f:
        torch.save(training_set, f)


if __name__ == '__main__':
    ds = FashionMNISTDataset()
    # 做类异常
    ds.gen_cls_attr(anom='005_100', feat1='ori', feat2='lbp')
    ds.gen_cls_attr(anom='005_100', feat1='ori', feat2='glcm')
    ds.gen_cls_attr(anom='005_100', feat1='lbp', feat2='glcm')

