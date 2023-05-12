# @Author  : Edlison
# @Date    : 6/7/21 20:16
# @Author  : Edlison
# @Date    : 5/5/21 12:17
# @Author  : Edlison
# @Date    : 5/5/21 11:17
import sys
sys.path.append('/home/rain/Projects/MVADG')

import argparse
import os
from torch.autograd import Variable
import torch
from itertools import chain as ichain
from v6.clusgan.models import MultiNet_mnist
from v6.clusgan.utils import calc_gradient_penalty, sample_z
from v6.complex.utils import dataloader_v2, dataloader_v3, save_net, draw_loss, draw_auc, draw_loss3v
from v6.complex.test import eval_enc_epoch, eval_gen_epoch, eval_enc_epoch3v
from tqdm import tqdm

torch.backends.cudnn.enabled = False  # 关闭cudnn


def run(device_id, wass_metric, n_epochs, batch_size, feat1, feat2, anom, lr1, lr2, b1, b2, decay, n_skip_iter,
        latent_dim, n_c, betan1, betan2, betac1, betac2, dataset):
    mtype = 'van'
    if wass_metric:
        mtype = 'wass'

    sep_und = '_'
    run_name_comps = [feat1, feat2, anom, mtype, 'e%i' % n_epochs, 'b%i' % batch_size, 'l%i' % latent_dim,
                      'betan1%.3f' % betan1, 'betac1%.3f' % betac1, 'betan2%.3f' % betan2, 'betac2%.3f' % betac2]
    run_name = sep_und.join(run_name_comps)

    # 生成输出的文件夹路径
    run_dir = os.path.join('./out', run_name)  # ./out/iepoch
    os.makedirs(run_dir, exist_ok=True)

    print('\nResults to be saved in directory %s\n' % (run_dir))

    # x_shape = (channels, img_size, img_size)
    x_shape = (1, 28, 28)

    cuda = True if torch.cuda.is_available() else False
    if cuda: torch.cuda.set_device(device_id)

    bce_loss1 = torch.nn.BCELoss()
    bce_loss2 = torch.nn.BCELoss()
    xe_loss1 = torch.nn.CrossEntropyLoss()
    xe_loss2 = torch.nn.CrossEntropyLoss()
    mse_loss1 = torch.nn.MSELoss()
    mse_loss2 = torch.nn.MSELoss()

    net = MultiNet_mnist(view_num=2, latent_dim=latent_dim, cls_num=n_c, x_shape=x_shape, wass=wass_metric)

    if cuda:
        net.cuda()
        bce_loss1.cuda()
        bce_loss2.cuda()
        xe_loss1.cuda()
        xe_loss2.cuda()
        mse_loss1.cuda()
        mse_loss2.cuda()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    loader = dataloader_v2(dataset=dataset, anom=anom, feat1=feat1, feat2=feat2, batch=batch_size, shuffle=True)

    GsEs_chain1 = ichain(net.Gs[0].parameters(), net.Es[0].parameters())
    GsEs_chain2 = ichain(net.Gs[1].parameters(), net.Es[1].parameters())
    optimizer_GsEs1 = torch.optim.Adam(GsEs_chain1, lr=lr1, betas=(b1, b2), weight_decay=decay)
    optimizer_GsEs2 = torch.optim.Adam(GsEs_chain2, lr=lr2, betas=(b1, b2), weight_decay=decay)
    optimizer_Ds1 = torch.optim.Adam(net.Ds[0].parameters(), lr=lr1, betas=(b1, b2))
    optimizer_Ds2 = torch.optim.Adam(net.Ds[1].parameters(), lr=lr2, betas=(b1, b2))

    eval_ge_loss1 = []
    eval_ge_loss2 = []
    eval_d_loss1 = []
    eval_d_loss2 = []
    eval_auc = []

    print('\nBegin training session with %i epochs...\n' % n_epochs)
    for epoch in tqdm(range(n_epochs)):
        for i, (data1, data2, labels) in tqdm(enumerate(loader)):
            net.Gs.train()
            net.Es.train()
            net.Gs.zero_grad()
            net.Es.zero_grad()
            net.Ds.zero_grad()

            real_imgs = [data1.type(Tensor), data2.type(Tensor)]

            # ---------------------------
            #  Train Generator + Encoder
            # ---------------------------
            optimizer_GsEs1.zero_grad()
            optimizer_GsEs2.zero_grad()
            zn, zc, zc_idx = sample_z(shape=data1.shape[0], latent_dim=latent_dim, n_c=n_c)
            zn = [zn for _ in range(2)]
            zc = [zc for _ in range(2)]
            zc_idx = [zc_idx for _ in range(2)]
            gen_imgs = net(zn=zn, zc=zc, mode='G')
            D_gen = net(data=gen_imgs, mode='D')
            D_real = net(data=real_imgs, mode='D')
            if (i % n_skip_iter == 0):
                encs = net(data=gen_imgs, mode='E')
                # enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encs
                zn_loss1 = mse_loss1(encs[0][0], zn[0])
                zc_loss1 = xe_loss1(encs[0][2], zc_idx[0])
                zn_loss2 = mse_loss2(encs[1][0], zn[1])
                zc_loss2 = xe_loss2(encs[1][2], zc_idx[1])
                if wass_metric:
                    ge_loss1 = torch.mean(D_gen[0]) + betan1 * zn_loss1 + betac1 * zc_loss1
                    ge_loss2 = torch.mean(D_gen[1]) + betan2 * zn_loss2 + betac2 * zc_loss2
                else:
                    valid1 = Variable(Tensor(gen_imgs[0].size(0), 1).fill_(1.0), requires_grad=False)
                    valid2 = Variable(Tensor(gen_imgs[1].size(0), 1).fill_(1.0), requires_grad=False)
                    v_loss1 = bce_loss1(D_gen[0], valid1)
                    v_loss2 = bce_loss2(D_gen[1], valid2)
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
                real_loss1 = bce_loss1(D_real[0], valid1)
                real_loss2 = bce_loss2(D_real[1], valid2)
                fake_loss1 = bce_loss1(D_gen[0], fake1)
                fake_loss2 = bce_loss2(D_gen[1], fake2)
                d_loss1 = (real_loss1 + fake_loss1) / 2
                d_loss2 = (real_loss2 + fake_loss2) / 2
            d_loss1.backward()
            d_loss2.backward()
            optimizer_Ds1.step()
            optimizer_Ds2.step()
            print("[Epoch {}/{}]\nLosses: [GE1: {}, GE2: {}] [D1: {}, D2: {}]".format(
                epoch, n_epochs, ge_loss1.item(), ge_loss2.item(), d_loss1.item(), d_loss2.item()))
        # ----
        # each epoch
        # ----
        eval_ge_loss1.append(ge_loss1)
        eval_ge_loss2.append(ge_loss2)
        eval_d_loss1.append(d_loss1)
        eval_d_loss2.append(d_loss2)
        if anom and (epoch + 1) % 1 == 0:
            auc = eval_enc_epoch(net=net, feat1=feat1, feat2=feat2, anom=anom)  # 非常慢1min
            eval_auc.append(auc)
        # if (epoch + 1) % 10 == 0:
        #     eval_gen_epoch(net=net, dir=run_dir, epoch=(epoch + 1))
    # ----
    # after epoch
    # ----
    save_net(net, run_dir)
    draw_loss(eval_ge_loss1, eval_ge_loss2, eval_d_loss1, eval_d_loss2, run_dir)
    draw_auc(eval_auc, run_dir)


def run_continue(view_num, latent_dim, cls_num, epochs, new_epochs, batch_size, feat1, feat2, anom, wass_metric, dataset):
    mtype = 'van'
    if wass_metric:
        mtype = 'wass'

    sep_und = '_'
    run_name_comps = [feat1, feat2, anom, mtype, 'e%i' % n_epochs, 'b%i' % batch_size, 'l%i' % latent_dim,
                      'betan1%.3f' % betan1, 'betac1%.3f' % betac1, 'betan2%.3f' % betan2, 'betac2%.3f' % betac2]
    new_run_name_comps = [feat1, feat2, anom, mtype, 'e%i' % (epochs + new_epochs), 'b%i' % batch_size, 'l%i' % latent_dim,
                      'betan1%.3f' % betan1, 'betac1%.3f' % betac1, 'betan2%.3f' % betan2, 'betac2%.3f' % betac2]
    run_name = sep_und.join(run_name_comps)
    new_run_name = sep_und.join(new_run_name_comps)
    net_dir = os.path.join('./out', run_name, 'net.pt')  # ./out/iepoch
    new_dir = os.path.join('./out', new_run_name)
    os.makedirs(new_dir, exist_ok=True)

    print('\nResults to be saved in directory %s\n' % (new_dir))

    x_shape = (1, 28, 28)

    net = MultiNet_mnist(view_num=view_num, latent_dim=latent_dim, cls_num=cls_num, x_shape=x_shape, wass=wass_metric)
    net.load_state_dict(torch.load(net_dir))

    cuda = True if torch.cuda.is_available() else False
    if cuda: torch.cuda.set_device(device_id)

    bce_loss1 = torch.nn.BCELoss()
    bce_loss2 = torch.nn.BCELoss()
    xe_loss1 = torch.nn.CrossEntropyLoss()
    xe_loss2 = torch.nn.CrossEntropyLoss()
    mse_loss1 = torch.nn.MSELoss()
    mse_loss2 = torch.nn.MSELoss()

    if cuda:
        net.cuda()
        bce_loss1.cuda()
        bce_loss2.cuda()
        xe_loss1.cuda()
        xe_loss2.cuda()
        mse_loss1.cuda()
        mse_loss2.cuda()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    loader = dataloader_v2(dataset=dataset, anom=anom, feat1=feat1, feat2=feat2, batch=batch_size, shuffle=True)

    GsEs_chain1 = ichain(net.Gs[0].parameters(), net.Es[0].parameters())
    GsEs_chain2 = ichain(net.Gs[1].parameters(), net.Es[1].parameters())
    optimizer_GsEs1 = torch.optim.Adam(GsEs_chain1, lr=lr1, betas=(b1, b2), weight_decay=decay)
    optimizer_GsEs2 = torch.optim.Adam(GsEs_chain2, lr=lr2, betas=(b1, b2), weight_decay=decay)
    optimizer_Ds1 = torch.optim.Adam(net.Ds[0].parameters(), lr=lr1, betas=(b1, b2))
    optimizer_Ds2 = torch.optim.Adam(net.Ds[1].parameters(), lr=lr2, betas=(b1, b2))

    eval_ge_loss1 = []
    eval_ge_loss2 = []
    eval_d_loss1 = []
    eval_d_loss2 = []
    eval_auc = []

    print('\nBegin training session from former net with %i epochs...\n' % new_epochs)
    for epoch in tqdm(range(new_epochs)):
        for i, (data1, data2, labels) in tqdm(enumerate(loader)):
            net.Gs.train()
            net.Es.train()
            net.Gs.zero_grad()
            net.Es.zero_grad()
            net.Ds.zero_grad()

            real_imgs = [data1.type(Tensor), data2.type(Tensor)]

            # ---------------------------
            #  Train Generator + Encoder
            # ---------------------------
            optimizer_GsEs1.zero_grad()
            optimizer_GsEs2.zero_grad()
            zn, zc, zc_idx = sample_z(shape=data1.shape[0], latent_dim=latent_dim, n_c=n_c)
            zn = [zn for _ in range(2)]
            zc = [zc for _ in range(2)]
            zc_idx = [zc_idx for _ in range(2)]
            gen_imgs = net(zn=zn, zc=zc, mode='G')
            D_gen = net(data=gen_imgs, mode='D')
            D_real = net(data=real_imgs, mode='D')
            if (i % n_skip_iter == 0):
                encs = net(data=gen_imgs, mode='E')
                # enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encs
                zn_loss1 = mse_loss1(encs[0][0], zn[0])
                zc_loss1 = xe_loss1(encs[0][2], zc_idx[0])
                zn_loss2 = mse_loss2(encs[1][0], zn[1])
                zc_loss2 = xe_loss2(encs[1][2], zc_idx[1])
                if wass_metric:
                    ge_loss1 = torch.mean(D_gen[0]) + betan1 * zn_loss1 + betac1 * zc_loss1
                    ge_loss2 = torch.mean(D_gen[1]) + betan2 * zn_loss2 + betac2 * zc_loss2
                else:
                    valid1 = Variable(Tensor(gen_imgs[0].size(0), 1).fill_(1.0), requires_grad=False)
                    valid2 = Variable(Tensor(gen_imgs[1].size(0), 1).fill_(1.0), requires_grad=False)
                    v_loss1 = bce_loss1(D_gen[0], valid1)
                    v_loss2 = bce_loss2(D_gen[1], valid2)
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
                real_loss1 = bce_loss1(D_real[0], valid1)
                real_loss2 = bce_loss2(D_real[1], valid2)
                fake_loss1 = bce_loss1(D_gen[0], fake1)
                fake_loss2 = bce_loss2(D_gen[1], fake2)
                d_loss1 = (real_loss1 + fake_loss1) / 2
                d_loss2 = (real_loss2 + fake_loss2) / 2
            d_loss1.backward()
            d_loss2.backward()
            optimizer_Ds1.step()
            optimizer_Ds2.step()
            print("[Epoch {}/{}]\nLosses: [GE1: {}, GE2: {}] [D1: {}, D2: {}]".format(
                epoch, new_epochs, ge_loss1.item(), ge_loss2.item(), d_loss1.item(), d_loss2.item()))
        # ----
        # each epoch
        # ----
        eval_ge_loss1.append(ge_loss1)
        eval_ge_loss2.append(ge_loss2)
        eval_d_loss1.append(d_loss1)
        eval_d_loss2.append(d_loss2)
        if anom and (epoch + 1) % 20 == 0:
            auc = eval_enc_epoch(net=net, feat1=feat1, feat2=feat2, anom=anom)  # 非常慢1min
            print('Auc: {}'.format(auc))
            eval_auc.append(auc)
        if (epoch + 1) % 10 == 0:
            eval_gen_epoch(net=net, dir=new_dir, epoch=(epoch + 1))
    # ----
    # after epoch
    # ----
    save_net(net, new_dir)
    draw_loss(eval_ge_loss1, eval_ge_loss2, eval_d_loss1, eval_d_loss2, new_dir)
    draw_auc(eval_auc, new_dir)


def run3v(device_id, wass_metric, n_epochs, batch_size, feat1, feat2, feat3, anom, lr1, lr2, lr3, b1, b2, decay, n_skip_iter,
        latent_dim, n_c, betan1, betan2, betan3, betac1, betac2, betac3, dataset):
    mtype = 'van'
    if wass_metric:
        mtype = 'wass'

    sep_und = '_'
    run_name_comps = [feat1, feat2, feat3, anom, mtype, 'e%i' % n_epochs, 'b%i' % batch_size, 'l%i' % latent_dim,
                      'betan1%.3f' % betan1, 'betac1%.3f' % betac1, 'betan2%.3f' % betan2, 'betac2%.3f' % betac2]
    run_name = sep_und.join(run_name_comps)

    # 生成输出的文件夹路径
    run_dir = os.path.join('./out', run_name)  # ./out/iepoch
    os.makedirs(run_dir, exist_ok=True)

    print('\nResults to be saved in directory %s\n' % (run_dir))

    # x_shape = (channels, img_size, img_size)
    x_shape = (1, 28, 28)

    cuda = True if torch.cuda.is_available() else False
    if cuda: torch.cuda.set_device(device_id)

    bce_loss1 = torch.nn.BCELoss()
    bce_loss2 = torch.nn.BCELoss()
    bce_loss3 = torch.nn.BCELoss()
    xe_loss1 = torch.nn.CrossEntropyLoss()
    xe_loss2 = torch.nn.CrossEntropyLoss()
    xe_loss3 = torch.nn.CrossEntropyLoss()
    mse_loss1 = torch.nn.MSELoss()
    mse_loss2 = torch.nn.MSELoss()
    mse_loss3 = torch.nn.MSELoss()

    net = MultiNet_mnist(view_num=3, latent_dim=latent_dim, cls_num=n_c, x_shape=x_shape, wass=wass_metric)

    if cuda:
        net.cuda()
        bce_loss1.cuda()
        bce_loss2.cuda()
        bce_loss3.cuda()
        xe_loss1.cuda()
        xe_loss2.cuda()
        xe_loss3.cuda()
        mse_loss1.cuda()
        mse_loss2.cuda()
        mse_loss3.cuda()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    loader = dataloader_v3(dataset=dataset, anom=anom, feat1=feat1, feat2=feat2, feat3=feat3, batch=batch_size, shuffle=True)

    GsEs_chain1 = ichain(net.Gs[0].parameters(), net.Es[0].parameters())
    GsEs_chain2 = ichain(net.Gs[1].parameters(), net.Es[1].parameters())
    GsEs_chain3 = ichain(net.Gs[2].parameters(), net.Es[2].parameters())
    optimizer_GsEs1 = torch.optim.Adam(GsEs_chain1, lr=lr1, betas=(b1, b2), weight_decay=decay)
    optimizer_GsEs2 = torch.optim.Adam(GsEs_chain2, lr=lr2, betas=(b1, b2), weight_decay=decay)
    optimizer_GsEs3 = torch.optim.Adam(GsEs_chain3, lr=lr3, betas=(b1, b2), weight_decay=decay)
    optimizer_Ds1 = torch.optim.Adam(net.Ds[0].parameters(), lr=lr1, betas=(b1, b2))
    optimizer_Ds2 = torch.optim.Adam(net.Ds[1].parameters(), lr=lr2, betas=(b1, b2))
    optimizer_Ds3 = torch.optim.Adam(net.Ds[2].parameters(), lr=lr3, betas=(b1, b2))

    eval_ge_loss1 = []
    eval_ge_loss2 = []
    eval_ge_loss3 = []
    eval_d_loss1 = []
    eval_d_loss2 = []
    eval_d_loss3 = []
    eval_auc = []

    print('\nBegin training session with %i epochs...\n' % n_epochs)
    for epoch in tqdm(range(n_epochs)):
        for i, (data1, data2, data3, labels) in tqdm(enumerate(loader)):
            net.Gs.train()
            net.Es.train()
            net.Gs.zero_grad()
            net.Es.zero_grad()
            net.Ds.zero_grad()

            real_imgs = [data1.type(Tensor), data2.type(Tensor), data3.type(Tensor)]

            # ---------------------------
            #  Train Generator + Encoder
            # ---------------------------
            optimizer_GsEs1.zero_grad()
            optimizer_GsEs2.zero_grad()
            optimizer_GsEs3.zero_grad()
            zn, zc, zc_idx = sample_z(shape=data1.shape[0], latent_dim=latent_dim, n_c=n_c)
            zn = [zn for _ in range(3)]
            zc = [zc for _ in range(3)]
            zc_idx = [zc_idx for _ in range(3)]
            gen_imgs = net(zn=zn, zc=zc, mode='G')
            D_gen = net(data=gen_imgs, mode='D')
            D_real = net(data=real_imgs, mode='D')
            if (i % n_skip_iter == 0):
                encs = net(data=gen_imgs, mode='E')
                # enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encs
                zn_loss1 = mse_loss1(encs[0][0], zn[0])
                zc_loss1 = xe_loss1(encs[0][2], zc_idx[0])
                zn_loss2 = mse_loss2(encs[1][0], zn[1])
                zc_loss2 = xe_loss2(encs[1][2], zc_idx[1])
                zn_loss3 = mse_loss2(encs[2][0], zn[2])
                zc_loss3 = xe_loss2(encs[2][2], zc_idx[2])
                if wass_metric:
                    ge_loss1 = torch.mean(D_gen[0]) + betan1 * zn_loss1 + betac1 * zc_loss1
                    ge_loss2 = torch.mean(D_gen[1]) + betan2 * zn_loss2 + betac2 * zc_loss2
                    ge_loss3 = torch.mean(D_gen[2]) + betan3 * zn_loss3 + betac3 * zc_loss3
                else:
                    valid1 = Variable(Tensor(gen_imgs[0].size(0), 1).fill_(1.0), requires_grad=False)
                    valid2 = Variable(Tensor(gen_imgs[1].size(0), 1).fill_(1.0), requires_grad=False)
                    valid3 = Variable(Tensor(gen_imgs[2].size(0), 1).fill_(1.0), requires_grad=False)
                    v_loss1 = bce_loss1(D_gen[0], valid1)
                    v_loss2 = bce_loss2(D_gen[1], valid2)
                    v_loss3 = bce_loss2(D_gen[2], valid2)
                    ge_loss1 = v_loss1 + betan1 * zn_loss1 + betac1 * zc_loss1
                    ge_loss2 = v_loss2 + betan2 * zn_loss2 + betac2 * zc_loss2
                    ge_loss3 = v_loss3 + betan3 * zn_loss3 + betac3 * zc_loss3
                ge_loss1.backward(retain_graph=True)
                ge_loss2.backward(retain_graph=True)
                ge_loss3.backward(retain_graph=True)
                optimizer_GsEs1.step()
                optimizer_GsEs2.step()
                optimizer_GsEs3.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_Ds1.zero_grad()
            optimizer_Ds2.zero_grad()
            optimizer_Ds3.zero_grad()
            if wass_metric:
                grad_penalty1 = calc_gradient_penalty(net.Ds[0], real_imgs[0], gen_imgs[0])
                grad_penalty2 = calc_gradient_penalty(net.Ds[1], real_imgs[1], gen_imgs[1])
                grad_penalty3 = calc_gradient_penalty(net.Ds[2], real_imgs[2], gen_imgs[2])
                d_loss1 = torch.mean(D_real[0]) - torch.mean(D_gen[0]) + grad_penalty1
                d_loss2 = torch.mean(D_real[1]) - torch.mean(D_gen[1]) + grad_penalty2
                d_loss3 = torch.mean(D_real[2]) - torch.mean(D_gen[2]) + grad_penalty3
            else:
                fake1 = Variable(Tensor(gen_imgs[0].size(0), 1).fill_(0.0), requires_grad=False)
                fake2 = Variable(Tensor(gen_imgs[1].size(0), 1).fill_(0.0), requires_grad=False)
                fake3 = Variable(Tensor(gen_imgs[2].size(0), 1).fill_(0.0), requires_grad=False)
                real_loss1 = bce_loss1(D_real[0], valid1)
                real_loss2 = bce_loss2(D_real[1], valid2)
                real_loss3 = bce_loss3(D_real[2], valid3)
                fake_loss1 = bce_loss1(D_gen[0], fake1)
                fake_loss2 = bce_loss2(D_gen[1], fake2)
                fake_loss3 = bce_loss3(D_gen[2], fake3)
                d_loss1 = (real_loss1 + fake_loss1) / 2
                d_loss2 = (real_loss2 + fake_loss2) / 2
                d_loss3 = (real_loss3 + fake_loss3) / 2
            d_loss1.backward()
            d_loss2.backward()
            d_loss3.backward()
            optimizer_Ds1.step()
            optimizer_Ds2.step()
            optimizer_Ds3.step()
            print("[Epoch {}/{}]\nLosses: [GE1: {}, GE2: {}, GE3: {}] [D1: {}, D2: {}, D3: {}]".format(
                epoch, n_epochs, ge_loss1.item(), ge_loss2.item(), ge_loss3.item(), d_loss1.item(), d_loss2.item(), d_loss3.item()))
        # ----
        # each epoch
        # ----
        eval_ge_loss1.append(ge_loss1)
        eval_ge_loss2.append(ge_loss2)
        eval_ge_loss3.append(ge_loss3)
        eval_d_loss1.append(d_loss1)
        eval_d_loss2.append(d_loss2)
        eval_d_loss3.append(d_loss3)
        # if anom and (epoch + 1) % 20 == 0:
        #     auc = eval_enc_epoch3v(net=net, feat1=feat1, feat2=feat2, anom=anom)  # 非常慢1min
        #     print('Auc: {}'.format(auc))
        #     eval_auc.append(auc)
        # if (epoch + 1) % 10 == 0:
        #     eval_gen_epoch(net=net, dir=run_dir, epoch=(epoch + 1))
    # ----
    # after epoch
    # ----
    save_net(net, run_dir)
    draw_loss3v(eval_ge_loss1, eval_ge_loss2, eval_ge_loss3, eval_d_loss1, eval_d_loss2, eval_d_loss3, run_dir)
    # draw_auc(eval_auc, run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST complex")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=1, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("-w", "--wass_metric", dest="wass_metric", action='store_true',
                        help="Flag for Wasserstein metric")
    parser.add_argument("-g", "-–gpu", dest="gpu", default=1, type=int, help="GPU id to use")
    parser.add_argument("-c", "--conti", dest="conti", action="store_true", help="If train from former net default false")
    args = parser.parse_args()

    # Training details
    device_id = args.gpu  # GPU编号
    wass_metric = args.wass_metric  # 是否使用W-GAN
    n_epochs = args.n_epochs  # epoch
    batch_size = args.batch_size  # batch
    c = args.conti  # 是否继续训练
    dataset = 'mnist'
    feat1 = 'ori'  # view1特征种类
    feat2 = 'lbp'  # view2特征种类
    feat3 = ''  # view3特征种类
    anom = '005_100'  # 异常比例 {'015', '005', '001', '005_0', '005_20', '005_40', '005_60', '005_80', '005_100'}
    lr1 = 1e-4  # 步长
    lr2 = 1e-4  # 步长
    lr3 = 1e-4  # 步长
    b1 = 0.5  # 优化器参数
    b2 = 0.9  # 优化器参数
    decay = 2.5 * 1e-5  # 优化器参数
    n_skip_iter = 1  # 训练n次D 训练1次G+E

    # Latent space info
    view_num = 2
    latent_dim = 30  # 用于生成图片的隐空间向量dim
    n_c = 10  # 聚类个数
    betan1 = 10  # 计算zn1损失时对应的参数
    betan2 = 10  # 计算zn2损失时对应的参数
    betan3 = 10  # 计算zn2损失时对应的参数
    betac1 = 10  # 计算zc1损失时对应的参数
    betac2 = 10  # 计算zc2损失时对应的参数
    betac3 = 10  # 计算zc2损失时对应的参数

    # Continue Train info
    new_epochs = 1

    if feat3 != '':
        run3v(device_id=device_id, wass_metric=wass_metric, n_epochs=n_epochs, batch_size=batch_size, feat1=feat1, feat2=feat2,
              feat3=feat3, anom=anom, lr1=lr1, lr2=lr2, lr3=lr3, b1=b1, b2=b2, decay=decay, n_skip_iter=n_skip_iter, latent_dim=latent_dim,
              n_c=n_c, betan1=betan1, betan2=betan2, betan3=betan3, betac1=betac1, betac2=betac2, betac3=betac3, dataset=dataset)
        exit(0)

    print('continue: ', c)
    if not c:
        print('This is new training.')
        run(device_id=device_id, wass_metric=wass_metric, n_epochs=n_epochs, batch_size=batch_size, feat1=feat1, feat2=feat2,
            anom=anom, lr1=lr1, lr2=lr2, b1=b1, b2=b2, decay=decay, n_skip_iter=n_skip_iter, latent_dim=latent_dim, n_c=n_c,
            betan1=betan1, betan2=betan2, betac1=betac1, betac2=betac2, dataset=dataset)
    else:
        print('This is continuing training.')
        run_continue(view_num=view_num, latent_dim=latent_dim, cls_num=n_c, epochs=n_epochs, new_epochs=new_epochs,
                     batch_size=batch_size, feat1=feat1, feat2=feat2, anom=anom, wass_metric=wass_metric, dataset=dataset)
