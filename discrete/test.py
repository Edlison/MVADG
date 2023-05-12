# @Author  : Edlison
# @Date    : 5/11/21 19:40
import torch
import numpy as np
from torch.utils.data import DataLoader
from v5.Dataset import ZooDataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from v6.discrete.utils import ZooDataset, sample_z, features2argmax, softmax2argmax, WineDataset
from v6.discrete.models import Generator, Discriminator, Encoder
from v5.Metrics import purity_score
from v5.Anomaly import roc_auc_score
import matplotlib.pyplot as plt
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def vote_score(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)
    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return y_true, y_voted_labels


def eval_generator(batch):
    latent_dim = 30
    cls_num = 3
    features_dim = 13
    generator = Generator(latent_dim, cls_num, features_dim)
    f_gen = './out/models/Generator.pt'
    generator.load_state_dict(torch.load(f_gen))
    generator.cuda()

    zn, zc, zc_idx = sample_z(batch, latent_dim, cls_num)
    gen = generator(zn, zc)
    # gen_max = features2argmax(gen)

    print(gen)


def eval_encoder(batch):
    latent_dim = 30
    cls_num = 3
    features_dim = 30
    features_dim = 13
    encoder = Encoder(latent_dim, cls_num, features_dim)
    f_enc = './out/models/Encoder.pt'
    encoder.load_state_dict(torch.load(f_enc))
    encoder.cuda()

    loader_zoo = DataLoader(ZooDataset(), batch_size=batch, shuffle=False)
    # loader_wine = DataLoader(WineDataset(), batch_size=batch, shuffle=False)
    data, labels = next(iter(loader_zoo))

    enc_zn, enc_zc, enc_zc_logits = encoder(data.type(Tensor))
    enc_labels = softmax2argmax(enc_zc)

    print('fake labels: ', enc_labels)
    print('ture labels: ', labels)
    print('purity score: ', purity_score(labels.numpy(), enc_labels.cpu().numpy()))


def eval_encoder_multi(batch):
    latent_dim = 30
    cls_num = 3
    features_dim = 13
    encoder1 = Encoder(latent_dim, cls_num, 7)
    encoder2 = Encoder(latent_dim, cls_num, 6)
    f_enc1 = './out/models/Encoder1.pt'
    f_enc2 = './out/models/Encoder2.pt'
    encoder1.load_state_dict(torch.load(f_enc1))
    encoder2.load_state_dict(torch.load(f_enc2))
    encoder1.cuda()
    encoder2.cuda()

    loader_zoo = DataLoader(ZooDataset(), batch_size=batch, shuffle=False)
    loader_wine = DataLoader(WineDataset(anom=True), batch_size=batch, shuffle=False)
    data1, data2, labels = next(iter(loader_wine))

    enc_zn, enc_zc1, enc_zc_logits = encoder1(data1.type(Tensor))
    enc_zn, enc_zc2, enc_zc_logits = encoder2(data2.type(Tensor))
    enc_labels1 = softmax2argmax(enc_zc1).cpu().numpy()
    enc_labels2 = softmax2argmax(enc_zc2).cpu().numpy()

    acc1 = purity_score(labels.numpy(), enc_labels1)
    acc2 = purity_score(labels.numpy(), enc_labels2)

    anom_score = np.zeros([batch, ])
    label1_map = {0:0, 2:2, 1:1}  # 需要两个label转换
    print(enc_labels1)
    print(enc_labels2)
    for i in range(batch):
        if label1_map[int(enc_labels1[i])] != enc_labels2[i]:
            anom_score[i] = 0.5
            # print(i, end=' ')
    print('auc ', roc_auc_score(labels, anom_score))


def eval_wine_epoch(encoder1, encoder2):
    encoder1.eval()
    encoder2.eval()
    loader_wine = DataLoader(WineDataset(anom=True), batch_size=135, shuffle=False)

    data1, data2, labels = next(iter(loader_wine))
    enc_zn1, enc_zc1, enc_zc_logits = encoder1(data1.type(Tensor))
    enc_zn2, enc_zc2, enc_zc_logits = encoder2(data2.type(Tensor))
    enc_labels1 = softmax2argmax(enc_zc1).cpu().numpy()
    enc_labels2 = softmax2argmax(enc_zc2).cpu().numpy()

    # 计算attr score
    attr_score1 = attr_anom(enc_zn1.cpu().detach().numpy())
    attr_score2 = attr_anom(enc_zn2.cpu().detach().numpy())
    attr_score = attr_score1 + attr_score2

    # 计算cls score
    label_map = [{0: 0, 1: 1, 2: 2}, {0: 0, 1: 2, 2: 1}, {0: 1, 1: 0, 2: 2}, {0: 1, 1: 2, 2: 0}, {0: 2, 1: 0, 2: 1}, {0: 2, 1: 2, 2: 1}]
    res = []
    for l in range(len(label_map)):
        cls_score = np.zeros([len(labels), ])
        for i in range(len(labels)):
            if label_map[l][int(enc_labels1[i])] != enc_labels2[i]:
                cls_score[i] = 0.5  # 没影响吧
        anom_score = attr_score + cls_score
        auc_each = roc_auc_score(labels, anom_score)
        res.append(auc_each)
    res.sort(reverse=True)
    print('res ', res[0])
    return res[0]


def eval_zoo_epoch(encoder1, encoder2):
    encoder1.eval()
    encoder2.eval()
    loader_zoo = DataLoader(ZooDataset(anom=True), batch_size=101, shuffle=False)

    data1, data2, labels = next(iter(loader_zoo))
    enc_zn1, enc_zc1, enc_zc_logits = encoder1(data1.type(Tensor))
    enc_zn2, enc_zc2, enc_zc_logits = encoder2(data2.type(Tensor))
    enc_labels1 = softmax2argmax(enc_zc1).cpu().numpy()
    enc_labels2 = softmax2argmax(enc_zc2).cpu().numpy()

    # 计算attr score
    attr_score1 = attr_anom(enc_zn1.cpu().detach().numpy())
    attr_score2 = attr_anom(enc_zn2.cpu().detach().numpy())
    attr_score = attr_score1 + attr_score2

    # 计算cls score
    voted_label1, voted_label2 = vote_score(enc_labels1, enc_labels2)
    cls_score = np.zeros([len(labels), ])
    for i in range(len(voted_label1)):
        if voted_label1[i] != voted_label2[i]:
            cls_score[i] += 0.5
    anom_score = attr_score + cls_score
    auc = roc_auc_score(labels, anom_score)
    print('auc ', auc)

    return auc


def attr_anom(zn):
    nei = NearestNeighbors(n_neighbors=2).fit(zn)
    distance, _ = nei.kneighbors(zn)
    score = distance[:, 1]
    return score


def show_loss_auc(ge_loss1, ge_loss2, d_loss1, d_loss2, auc):
    epoch = range(0, len(ge_loss1))
    plt.title('loss & auc')
    plt.plot(epoch, ge_loss1, color='blue', label='ge_loss1')
    plt.plot(epoch, ge_loss2, color='skyblue', label='ge_loss2')
    plt.plot(epoch, d_loss1, color='green', label='d_loss1')
    plt.plot(epoch, d_loss2, color='orange', label='d_loss2')
    plt.plot(epoch, auc, color='red', label='auc')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss or auc')
    plt.savefig('loss_auc.png')
    plt.show()

    auc.sort(reverse=True)
    print('auc best: ', auc[0])


if __name__ == '__main__':
    # eval_encoder(batch=135)
    # eval_generator(batch=10)
    # eval_encoder_multi(batch=135)
    ds = ZooDataset(anom=True)
    loader = DataLoader(ds, batch_size=101, shuffle=False)
    data1, data2, labels = next(iter(loader))
    print(data1.shape)
    print(data2.shape)
    print(labels.shape)
