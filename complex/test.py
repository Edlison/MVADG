# @Author  : Edlison
# @Date    : 6/7/21 21:16
import os
import torch
import numpy as np
from v6.complex.utils import dataloader_v2, dataloader_v3
from v6.discrete.test import softmax2argmax, attr_anom, vote_score, roc_auc_score
from v6.clusgan.utils import sample_z
from v6.Eval import show
import matplotlib.pyplot as plt

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def eval_enc_epoch(net, dataset='mnist', feat1='ori', feat2='lbp', anom='005'):
    net_ = net
    net_.to('cpu')
    net_.eval()
    # if cuda:
    #     net.cuda()
    loader = dataloader_v2(dataset=dataset, anom=anom, feat1=feat1, feat2=feat2, batch=60000, shuffle=False)
    data1, data2, labels = next(iter(loader))
    data = (data1, data2)
    enc = net_(data=data, mode='E')
    enc_labels1 = softmax2argmax(enc[0][1]).cpu().numpy()
    enc_labels2 = softmax2argmax(enc[1][1]).cpu().numpy()

    from time import time
    # 计算attr score
    t1 = time()
    attr_score1 = attr_anom(enc[0][0].cpu().detach().numpy())
    attr_score2 = attr_anom(enc[1][0].cpu().detach().numpy())
    attr_score = attr_score1 + attr_score2
    t2 = time()
    # 计算cls score
    voted_label1, voted_label2 = vote_score(enc_labels1, enc_labels2)
    cls_score = np.zeros([len(labels), ])
    for i in range(len(voted_label1)):
        if voted_label1[i] != voted_label2[i]:
            cls_score[i] += 0.5
    anom_score = attr_score + cls_score
    auc = roc_auc_score(labels, anom_score)
    t3 = time()
    print('attr cost {}s, cls cost {}s.'.format((t2 - t1), (t3 - t2)))
    print('auc ', auc)
    return auc


def eval_enc_epoch3v(net, dataset='mnist', feat1='ori', feat2='lbp', feat3='glcm', anom='005'):
    print('start eval enc')
    net.eval()
    # if cuda:
    #     net.cuda()
    loader = dataloader_v3(dataset=dataset, anom=anom, feat1=feat1, feat2=feat2, feat3=feat3, batch=60000, shuffle=False)
    data1, data2, data3, labels = next(iter(loader))
    data = (data1, data2, data3)
    enc = net(data=data, mode='E')
    enc_labels1 = softmax2argmax(enc[0][1]).cpu().numpy()
    enc_labels2 = softmax2argmax(enc[1][1]).cpu().numpy()
    enc_labels3 = softmax2argmax(enc[2][1]).cpu().numpy()

    from time import time
    # 计算attr score
    t1 = time()
    attr_score1 = attr_anom(enc[0][0].cpu().detach().numpy())
    attr_score2 = attr_anom(enc[1][0].cpu().detach().numpy())
    attr_score3 = attr_anom(enc[2][0].cpu().detach().numpy())
    attr_score = attr_score1 + attr_score2 + attr_score3
    t2 = time()

    # 计算cls score
    voted_label1, voted_label2 = vote_score(enc_labels1, enc_labels2)
    _, voted_label3 = vote_score(voted_label2, enc_labels3)
    cls_score = np.zeros([len(labels), ])
    for i in range(len(voted_label1)):
        if not voted_label1[i] == voted_label2[i] == voted_label3[i]:
            cls_score[i] += 0.5
    anom_score = attr_score + cls_score
    auc = roc_auc_score(labels, anom_score)
    t3 = time()
    print('attr cost {}s, cls cost {}s.'.format((t2 - t1), (t3 - t2)))
    print('auc ', auc)
    return auc


def eval_gen_epoch(net, dir, epoch):
    net.eval()
    eval_num = 25
    zn, zc, _ = sample_z(shape=eval_num, latent_dim=net.latent_dim, n_c=net.cls_num)
    zn = [zn for _ in range(net.view_num)]
    zc = [zc for _ in range(net.view_num)]
    gen_imgs = net(zn=zn, zc=zc, mode='G')
    for view in range(net.view_num):
        for i in range(eval_num):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(gen_imgs[view][i].squeeze().cpu().detach().numpy())
        plt.title('view {}'.format(view))
        tar_f = os.path.join(dir, 'net{}_gen_{}.png'.format(view, epoch))
        plt.savefig(tar_f)
        plt.show()


if __name__ == '__main__':
    from v6.clusgan.models import MultiNet_mnist
    # Params
    view_num = 2
    latent_dim = 30
    cls_num = 10
    x_shape = (1, 28, 28)
    dataset = 'mnist'
    feat1 = 'ori'
    feat2 = 'lbp'
    feat3 = 'glcm'
    anom = '005_100'
    wass = True

    # Init net
    net_f = './out/ori_lbp_005_100_wass_e300_b32_l30_betan10.500_betac15.000_betan25.000_betac20.500/net.pt'
    net = MultiNet_mnist(view_num=view_num, latent_dim=latent_dim, cls_num=cls_num, x_shape=x_shape, wass=wass)
    net.load_state_dict(torch.load(net_f, map_location=lambda storage, loc: storage))

    # Eval net
    if view_num == 2:
        auc = eval_enc_epoch(net, dataset=dataset, feat1=feat1, feat2=feat2, anom=anom)
    elif view_num == 3:
        auc = eval_enc_epoch3v(net, dataset=dataset, feat1=feat1, feat2=feat2, feat3=feat3, anom=anom)
    else:
        raise ValueError(view_num, 'err')

