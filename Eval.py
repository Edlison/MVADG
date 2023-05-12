# @Author  : Edlison
# @Date    : 4/28/21 22:04
import sys

sys.path.append('/home/rain/Projects/MVADG/v6/')
import torch
from clusgan.models import MultiNet_mnist_usps, MultiNet_mnist, MultiNet_usps, Generator_Simple, Encoder_Simple,\
    MultiNet_simple, MultiNet_usps_simple
from clusgan.utils import sample_z
from clusgan.datasets import get_loader, get_anom_loader, get_multi_loader, get_multi_anom_loader, \
    MNISTAnomDataset, MNISTDataset, UspsAnomDataset, UspsDataset, gen_anom_cls, gen_anom_attr
from v5.Metrics import purity_score
from sklearn.cluster import KMeans
import numpy as np
from v5.Anomaly import anom_auc, anom_avg_distance, roc_auc_score
from v6.complex.utils import dataloader_v2


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


def gen_mnist(dir, latent_dim, cls_num, wass):
    f = './out/' + dir + '/models/net.pt'
    batch = 25

    net = MultiNet_usps(view_num=2, latent_dim=latent_dim, cls_num=cls_num, x_shape=(1, 16, 16), wass=wass)
    # net = MultiNet_simple(view_num=2, latent_dim=latent_dim, cls_num=cls_num, wass=wass)
    net.load_state_dict(torch.load(f))
    net.cuda()

    zn, zc, zc_idx = sample_z(batch, latent_dim=latent_dim, n_c=cls_num)
    zn = [zn for _ in range(2)]
    zc = [zc for _ in range(2)]

    gen_imgs = net(zn=zn, zc=zc, mode='G')

    temp1 = gen_imgs[0].cpu().detach().numpy()
    temp2 = gen_imgs[1].cpu().detach().numpy()
    show(temp1, 5, 5)
    show(temp2, 5, 5)


def enc_mnist(dir, processed, latent_dim, cls_num, wass):
    f = './out/' + dir + '/models/net.pt'
    batch = 25
    net = MultiNet_usps(view_num=2, latent_dim=latent_dim, cls_num=cls_num, x_shape=(1, 16, 16), wass=wass)
    # net = MultiNet_simple(view_num=2, latent_dim=latent_dim, cls_num=cls_num, wass=wass)
    net.load_state_dict(torch.load(f))

    loader = get_anom_loader(ds='mnist', processed=processed, batch_num=batch, shuffle=True)
    imgs, labels = next(iter(loader))
    imgs = imgs[-25:]
    show(imgs, 5, 5)

    imgs = [imgs for _ in range(2)]
    encs = net(data=imgs, mode='E')

    print(torch.argmax(encs[0][2], dim=1))
    print(torch.argmax(encs[1][2], dim=1))


def gen_usps(dir, latent_dim, cls_num, x_shape):
    f = './out/' + dir + '/models/net.pt'
    batch = 25

    wass = True
    net = MultiNet_usps(view_num=2, latent_dim=latent_dim, cls_num=cls_num, x_shape=x_shape, wass=wass)
    net.load_state_dict(torch.load(f))
    net.cuda()

    zn, zc, zc_idx = sample_z(batch, latent_dim=latent_dim, n_c=cls_num)
    zn = [zn for _ in range(2)]
    zc = [zc for _ in range(2)]

    gen_imgs = net(zn=zn, zc=zc, mode='G')

    temp1 = gen_imgs[0].cpu().detach().numpy()
    temp2 = gen_imgs[1].cpu().detach().numpy()
    show(temp1, 5, 5)
    show(temp2, 5, 5)


def enc_usps(dir, processed, latent_dim, cls_num, x_shape):
    f = './out/' + dir + '/models/net.pt'
    batch = 25
    wass = True
    net = MultiNet_usps(view_num=2, latent_dim=latent_dim, cls_num=cls_num, x_shape=x_shape, wass=wass)
    net.load_state_dict(torch.load(f))

    loader = get_anom_loader(ds='usps', processed=processed, batch_num=batch, shuffle=True)
    imgs, labels = next(iter(loader))
    imgs = imgs[-25:]
    show(imgs, 5, 5)

    imgs = [imgs for _ in range(2)]
    encs = net(data=imgs, mode='E')

    print(torch.argmax(encs[0][2], dim=1))
    print(torch.argmax(encs[1][2], dim=1))


def gen_mnist_usps(dir, latent_dim, cls_num, wass):
    f = './out/' + dir + '/models/net.pt'
    print('eval {}'.format(f))
    batch = 25
    x_shape2 = (1, 16, 16)
    net = MultiNet_usps_simple(view_num=2, latent_dim=latent_dim, cls_num=cls_num, wass=wass)
    # net = MultiNet_simple(view_num=2, latent_dim=latent_dim, cls_num=cls_num, wass=wass)
    net.load_state_dict(torch.load(f))
    net.cuda()

    zn, zc, zc_idx = sample_z(batch, latent_dim=latent_dim, n_c=cls_num)
    zn = [zn for _ in range(2)]
    zc = [zc for _ in range(2)]

    gen_imgs = net(zn=zn, zc=zc, mode='G')

    temp1 = gen_imgs[0].cpu().detach().numpy()
    temp2 = gen_imgs[1].cpu().detach().numpy()
    show(temp1, 5, 5)
    show(temp2, 5, 5)


def enc_mnist_usps(dir, latent_dim, cls_num, wass):
    f = './out/' + dir + '/models/net.pt'
    print('eval {}'.format(f))
    batch = 1500
    x_shape2 = (1, 16, 16)
    net = MultiNet_usps_simple(view_num=2, latent_dim=latent_dim, cls_num=cls_num, wass=wass)
    # net = MultiNet_simple(view_num=2, latent_dim=latent_dim, cls_num=cls_num, wass=wass)
    net.load_state_dict(torch.load(f))

    loader = get_multi_loader(batch=batch, shuffle=True)
    mnist, usps = next(iter(loader))
    (mnist_imgs, mnist_labels), (usps_imgs, usps_labels) = mnist, usps
    mnist_labels = np.array(mnist_labels, dtype=np.int)
    usps_labels = np.array(usps_labels, dtype=np.int)
    imgs = [mnist_imgs, usps_imgs]

    show(mnist_imgs[:25], 5, 5)
    show(usps_imgs[:25], 5, 5)

    encs = net(data=imgs, mode='E')

    print('arg max')
    mnist_am = torch.argmax(encs[0][1], dim=1)
    usps_am = torch.argmax(encs[1][1], dim=1)
    print(mnist_am)
    print(usps_am)
    mnist_acc = purity_score(mnist_labels, np.array(mnist_am))
    usps_acc = purity_score(usps_labels, np.array(usps_am))
    print('mnist acc: ', mnist_acc)
    print('usps acc: ', usps_acc)
    print()

    # KMeans 效果不行！！！
    # print('kmeans zn')
    # km1 = KMeans(n_clusters=3, random_state=0).fit(encs[0][0].detach().numpy()).labels_  # zn + zc
    # km2 = KMeans(n_clusters=3, random_state=0).fit(encs[1][0].detach().numpy()).labels_
    # l1 = purity_score(mnist_labels, km1)
    # l2 = purity_score(usps_labels, km2)
    # print('km_label1: ', km1)
    # print('labels: ', mnist_labels)
    # print('km_label2: ', km2)
    # print('labels: ', usps_labels)
    # print('mnist acc: ', l1)
    # print('usps acc: ', l2)
    # print()
    #
    # print('kmeans zn+zc')
    # km1_ = KMeans(n_clusters=3, random_state=0).fit(torch.cat([encs[0][0], encs[0][1]], dim=1).detach().numpy()).labels_
    # km2_ = KMeans(n_clusters=3, random_state=0).fit(torch.cat([encs[1][0], encs[1][1]], dim=1).detach().numpy()).labels_
    # l1_ = purity_score(mnist_labels, km1_)
    # l2_ = purity_score(usps_labels, km2_)
    # print('mnist acc: ', l1_)
    # print('usps acc: ', l2_)


def gen_enc_mnist_usps(dir, latent_dim, cls_num, wass):
    f = './out/' + dir + '/models/net.pt'
    print('eval {}'.format(f))
    batch = 25
    x_shape2 = (1, 16, 16)
    net = MultiNet_usps_simple(view_num=2, latent_dim=latent_dim, cls_num=cls_num, wass=wass)
    # net = MultiNet_simple(view_num=2, latent_dim=latent_dim, cls_num=cls_num, wass=False)
    net.load_state_dict(torch.load(f))
    net.cuda()

    zn, zc, zc_idx = sample_z(batch, latent_dim=latent_dim, n_c=cls_num)
    zn = [zn for _ in range(2)]
    zc = [zc for _ in range(2)]

    gen_imgs = net(zn=zn, zc=zc, mode='G')
    temp1 = gen_imgs[0].cpu().detach().numpy()
    temp2 = gen_imgs[1].cpu().detach().numpy()
    print('gen res: ')
    show(temp1, 5, 5)
    show(temp2, 5, 5)

    encs = net(data=gen_imgs, mode='E')
    print('enc res: ')
    print(torch.argmax(encs[0][2], dim=1))
    print(torch.argmax(encs[1][2], dim=1))


def eval_simple(dir):
    batch = 25
    generator = Generator_Simple()
    generator.load_state_dict(torch.load(dir))
    generator.cuda()
    zn, zc, zc_idx = sample_z(batch, latent_dim=30, n_c=3)
    gen_imgs = generator(zn, zc)
    show(gen_imgs.cpu().detach().numpy(), 5, 5)


def eval_simple_enc(dir):
    batch = 1500
    encoder = Encoder_Simple()
    encoder.load_state_dict(torch.load(dir))
    loader = get_multi_loader(batch=batch, shuffle=True)
    mnist, usps = next(iter(loader))
    (usps_imgs, usps_labels) = usps
    enc_zn, enc_zc, enc_logits = encoder(usps_imgs)
    simple_argmax = torch.argmax(enc_zc, dim=1)
    acc = purity_score(usps_labels.numpy(), simple_argmax.numpy())
    print('acc: ', acc)


def ds_test():
    loader1 = get_anom_loader(ds='mnist', processed='anom_37_4000_4000.pt', batch_num=25, shuffle=False)
    loader2 = get_anom_loader(ds='mnist', processed='mnist_2_4000.pt', batch_num=25, shuffle=False)
    imgs, _ = next(iter(loader1))
    show(imgs[0], 1, 1)
    print(imgs[0].shape)
    print(imgs[0])
    imgs, _ = next(iter(loader2))
    show(imgs[0], 1, 1)
    print(imgs[0].shape)
    print(imgs[0])


def anom_test(dir, latent_dim, cls_num):
    f = './out/' + dir + '/models/net.pt'
    print('eval {}'.format(f))
    batch = 1500
    x_shape2 = (1, 16, 16)
    # net = MultiNet_usps(view_num=2, latent_dim=latent_dim, cls_num=cls_num, x_shape=x_shape2, wass=True)
    net = MultiNet_usps_simple(view_num=2, latent_dim=latent_dim, cls_num=cls_num, wass=True)
    net.load_state_dict(torch.load(f))

    loader = get_multi_anom_loader(batch=batch, shuffle=True)
    mnist, usps = next(iter(loader))
    (mnist_imgs, mnist_labels), (usps_imgs, usps_labels) = mnist, usps
    mnist_labels = np.array(mnist_labels, dtype=np.int)
    usps_labels = np.array(usps_labels, dtype=np.int)
    imgs = [mnist_imgs, usps_imgs]

    # show(mnist_imgs[:25], 5, 5)
    # show(usps_imgs[:25], 5, 5)

    encs = net(data=imgs, mode='E')

    print('arg max')
    mnist_am = torch.argmax(encs[0][1], dim=1)
    usps_am = torch.argmax(encs[1][1], dim=1)
    mnist_acc = purity_score(mnist_labels, np.array(mnist_am))
    usps_acc = purity_score(usps_labels, np.array(usps_am))
    print('mnist acc: ', mnist_acc)
    print('usps acc: ', usps_acc)
    print(mnist_am[:50])
    print(usps_am[:50])
    print()

    score_cls = np.zeros([batch, ])
    # TODO 通过看标签与生成图片的对应关系 对usps_am做一个映射
    mnist_map = {1:2, 0:1, 2:0}  # e800
    usps_map = {2:1, 0:0, 1:2}  # e2k
    for i in range(batch):
        # if mnist_am[i] != usps_map[int(usps_am[i])]:
        #     score_cls[i] = 0.5
        if mnist_map[int(mnist_am[i])] != usps_am[i]:
            score_cls[i] = 0.5

    score_attr = np.zeros([batch, ])
    # TODO 属性异常 计算zn距离

    score_all = score_cls + score_attr
    auc = roc_auc_score(mnist_labels, score_all)
    print('anom labels: ', mnist_labels[:150])
    print('anom score: ', score_all[:150])
    print(auc)


if __name__ == '__main__':
    dir = '500epoch_z30_wass_bs16_nc3'
    dir2 = '2usps_5000epoch_z30_wass_bs16_nc3'
    dir3 = '2mnist_200epoch_z30_van_bs50_nc3'
    dir4 = 'mnist_usps_500epoch_z30_wass_bs50_nc3'
    simple_gen = './out/simple_2usps_500epoch_z30_van_bs50_nc3/models/generator.pt'
    simple_enc = './out/simple_2usps_500epoch_z30_van_bs50_nc3/models/encoder.pt'
    latent_dim = 30
    cls_num = 3

    # gen_usps(dir2, latent_dim, cls_num, x_shape=(1, 16, 16))
    # enc_usps(dir2, 'usps_3_500.pt', latent_dim, cls_num, x_shape=(1, 16, 16))

    # gen_mnist(dir3, latent_dim, cls_num, wass=False)
    # enc_mnist(dir3, 'mnist_3_500.pt', latent_dim, cls_num, wass=False)
    # gen_enc_mnist_usps(dir3, latent_dim, cls_num)

    # gen_mnist_usps(dir4, latent_dim, cls_num, wass=True)
    # enc_mnist_usps(dir4, latent_dim, cls_num, wass=True)
    # gen_enc_mnist_usps(dir4, latent_dim, cls_num, wass=True)

    # eval_simple(latent_dim, cls_num, x_shape=(1, 16, 16))
    # anom_test(dir4, latent_dim, cls_num)

    # eval_simple(simple_gen)
    # eval_simple_enc(simple_enc)

    # loader = dataloader_v2(dataset='mnist', anom='', feat1='lbp', feat2='glcm', batch=25, shuffle=False)
    # data1, data2, _ = next(iter(loader))

    file = '../data/mnist_custom/processed/mnist_ori_glcm.pt'
    data1, data2, _ = torch.load(file)

    print(data1.shape)
    print(data2.shape)

    show(data1[:25], 5, 5)
    show(data2[:25], 5, 5)

