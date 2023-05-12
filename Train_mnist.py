# @Author  : Edlison
# @Date    : 5/5/21 12:17
# @Author  : Edlison
# @Date    : 5/5/21 11:17
import argparse
import os
import pandas as pd
from torch.autograd import Variable
import torch
from torchvision.utils import save_image
from itertools import chain as ichain
from clusgan.definitions import DATASETS_DIR, RUNS_DIR
from clusgan.models import Generator_CNN, Encoder_CNN, Discriminator_CNN, MultiNet_mnist, MultiNet_usps, MultiNet_simple
from clusgan.utils import save_model, calc_gradient_penalty, sample_z, cross_entropy
from clusgan.datasets import get_dataloader, dataset_list, get_anom_loader, get_multi_loader
from clusgan.plots import plot_train_loss
from tqdm import tqdm

torch.backends.cudnn.enabled = False  # 关闭cudnn


def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default='clusgan', help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=1, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,
                        help="Dataset name")
    parser.add_argument("-w", "--wass_metric", dest="wass_metric", action='store_true',
                        help="Flag for Wasserstein metric")
    parser.add_argument("-g", "-–gpu", dest="gpu", default=1, type=int, help="GPU id to use")
    parser.add_argument("-k", "-–num_workers", dest="num_workers", default=1, type=int,
                        help="Number of dataset workers")
    args = parser.parse_args()
    run_name = args.run_name
    dataset_name = args.dataset_name
    device_id = args.gpu
    num_workers = args.num_workers
    wass_metric = args.wass_metric  # 是否使用W-GAN
    model_name = '2mnist'

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr1 = 1e-4
    lr2 = 1e-4
    b1 = 0.5
    b2 = 0.9
    decay = 2.5 * 1e-5
    n_skip_iter = 1

    # Latent space info
    latent_dim = 30  # 用于生成图片的隐空间向量dim
    n_c = 3  # 聚类个数
    betan1 = 10  # 计算zn损失时对应的参数
    betan2 = 10
    betac1 = 10  # 计算zc损失时对应的参数
    betac2 = 10  # 计算zc损失时对应的参数

    mtype = 'van'
    if (wass_metric):
        mtype = 'wass'

    # processed = 'anom_37_4000_4000.pt'
    sep_und = '_'
    run_name_comps = ['%s' % model_name, '%iepoch' % n_epochs, 'z%i' % latent_dim, mtype, 'bs%i' % batch_size, 'nc%i' % n_c]
    run_name = sep_und.join(run_name_comps)

    # 生成输出的文件夹路径
    run_dir = os.path.join('./out', run_name)  # ./out/iepoch
    models_dir = os.path.join(run_dir, 'models')  # ./out/iepoch/models
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print('\nResults to be saved in directory %s\n' % (run_dir))

    # x_shape = (channels, img_size, img_size)
    x_shape1 = (1, 28, 28)
    x_shape2 = (1, 16, 16)

    cuda = True if torch.cuda.is_available() else False
    if cuda: torch.cuda.set_device(device_id)

    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    net = MultiNet_usps(view_num=2, latent_dim=latent_dim, cls_num=n_c, x_shape=x_shape2, wass=wass_metric)
    # net = MultiNet_simple(view_num=2, latent_dim=latent_dim, cls_num=n_c, wass=wass_metric)

    print(net)
    if cuda:
        net.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    dataloader = get_anom_loader(ds='mnist', processed='mnist_3_500.pt', batch_num=batch_size, shuffle=True)

    GsEs_chain1 = ichain(net.Gs[0].parameters(), net.Es[0].parameters())
    GsEs_chain2 = ichain(net.Gs[1].parameters(), net.Es[1].parameters())
    optimizer_GsEs1 = torch.optim.Adam(GsEs_chain1, lr=lr1, betas=(b1, b2), weight_decay=decay)
    optimizer_GsEs2 = torch.optim.Adam(GsEs_chain2, lr=lr2, betas=(b1, b2), weight_decay=decay)
    optimizer_Ds1 = torch.optim.Adam(net.Ds[0].parameters(), lr=lr1, betas=(b1, b2))
    optimizer_Ds2 = torch.optim.Adam(net.Ds[1].parameters(), lr=lr2, betas=(b1, b2))

    print('\nBegin training session with %i epochs...\n' % (n_epochs))
    for epoch in tqdm(range(n_epochs)):
        for i, (imgs, labels) in tqdm(enumerate(dataloader)):
            net.Gs.train()
            net.Es.train()
            net.Gs.zero_grad()
            net.Es.zero_grad()
            net.Ds.zero_grad()

            # real_imgs = [imgs.type(Tensor) for _ in range(2)]
            real_imgs = [imgs.type(Tensor), imgs.type(Tensor)]

            # ---------------------------
            #  Train Generator + Encoder
            # ---------------------------
            optimizer_GsEs1.zero_grad()
            optimizer_GsEs2.zero_grad()
            zn, zc, zc_idx = sample_z(shape=imgs.shape[0], latent_dim=latent_dim, n_c=n_c)
            zn = [zn for _ in range(2)]
            zc = [zc for _ in range(2)]
            zc_idx = [zc_idx for _ in range(2)]
            gen_imgs = net(zn=zn, zc=zc, mode='G')
            D_gen = net(data=gen_imgs, mode='D')
            D_real = net(data=real_imgs, mode='D')
            if (i % n_skip_iter == 0):
                encs = net(data=gen_imgs, mode='E')
                # enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encs
                zn_loss1 = mse_loss(encs[0][0], zn[0])
                zc_loss1 = xe_loss(encs[0][2], zc_idx[0])
                zn_loss2 = mse_loss(encs[1][0], zn[1])
                zc_loss2 = xe_loss(encs[1][2], zc_idx[1])
                if wass_metric:
                    ge_loss1 = torch.mean(D_gen[0]) + betan1 * zn_loss1 + betac1 * zc_loss1
                    ge_loss2 = torch.mean(D_gen[1]) + betan2 * zn_loss2 + betac2 * zc_loss2
                else:
                    valid1 = Variable(Tensor(gen_imgs[0].size(0), 1).fill_(1.0), requires_grad=False)
                    valid2 = Variable(Tensor(gen_imgs[1].size(0), 1).fill_(1.0), requires_grad=False)
                    v_loss1 = bce_loss(D_gen[0], valid1)
                    v_loss2 = bce_loss(D_gen[1], valid2)
                    ge_loss1 = v_loss1 + betan1 * zn_loss1 + betac1 * zc_loss1
                    ge_loss2 = v_loss2 + betan2 * zn_loss2 + betac2 * zc_loss2
                ge_loss1.backward(retain_graph=True)
                ge_loss2.backward(retain_graph=True)
                optimizer_GsEs1.step()
                optimizer_GsEs2.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_Ds1.zero_grad()
            optimizer_Ds2.zero_grad()
            if wass_metric:
                grad_penalty1 = calc_gradient_penalty(net.Ds[0], real_imgs[0], gen_imgs[0])
                grad_penalty2 = calc_gradient_penalty(net.Ds[1], real_imgs[1], gen_imgs[1])
                d_loss1 = torch.mean(D_real[0]) - torch.mean(D_gen[0]) + grad_penalty1
                d_loss2 = torch.mean(D_real[1]) - torch.mean(D_gen[1]) + grad_penalty2
            else:
                fake1 = Variable(Tensor(gen_imgs[0].size(0), 1).fill_(0.0), requires_grad=False)
                fake2 = Variable(Tensor(gen_imgs[1].size(0), 1).fill_(0.0), requires_grad=False)
                real_loss1 = bce_loss(D_real[0], valid1)
                real_loss2 = bce_loss(D_real[1], valid2)
                fake_loss1 = bce_loss(D_gen[0], fake1)
                fake_loss2 = bce_loss(D_gen[1], fake2)
                d_loss1 = (real_loss1 + fake_loss1) / 2
                d_loss2 = (real_loss2 + fake_loss2) / 2
            d_loss1.backward()
            d_loss2.backward()
            optimizer_Ds1.step()
            optimizer_Ds2.step()
        # ----
        # each epoch
        # ----
        print("[Epoch {}/{}] \n\tModel Losses: [D1: {}, D2: {}] [GE1: {}, GE2: {}]".format(
            epoch, n_epochs, d_loss1.item(), d_loss2.item(), ge_loss1.item(), ge_loss2.item()))
    # ----
    # after epoch
    # ----
    torch.save(net.state_dict(), os.path.join(models_dir, 'net.pt'))


if __name__ == "__main__":
    main()
