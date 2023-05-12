# @Author  : Edlison
# @Date    : 5/6/21 11:49
import argparse
import os
import pandas as pd
from torch.autograd import Variable
import torch
from torchvision.utils import save_image
from itertools import chain as ichain
from clusgan.definitions import DATASETS_DIR, RUNS_DIR
from clusgan.models import Generator_Simple, Encoder_Simple, Discriminator_Simple
from clusgan.utils import save_model, calc_gradient_penalty, sample_z, cross_entropy
from clusgan.datasets import get_dataloader, dataset_list, get_usps_loader, get_anom_loader
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
    wass_metric = args.wass_metric  # WGAN不收敛 VGAN梯度消失
    model_name = '2usps'
    net_name = 'simple'

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr1 = 0.0002 # 1e-4
    b1 = 0.5
    b2 = 0.999 # 0.9
    decay = 2e-5 # 2.5 * 1e-5
    n_skip_iter = 1

    # Latent space info
    latent_dim = 30  # 用于生成图片的隐空间向量dim
    n_c = 3  # 聚类个数
    betan1 = 10  # 计算zn损失时对应的参数
    betac1 = 10  # 计算zc损失时对应的参数

    mtype = 'van'
    if (wass_metric):
        mtype = 'wass'

    # processed = 'anom_37_4000_4000.pt'
    sep_und = '_'
    run_name_comps = ['%s' % net_name, '%s' % model_name, '%iepoch' % n_epochs, 'z%i' % latent_dim, mtype, 'bs%i' % batch_size, 'nc%i' % n_c]
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

    generator = Generator_Simple()
    discriminator = Discriminator_Simple(wass=wass_metric)
    encoder = Encoder_Simple()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        encoder.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    dataloader = get_anom_loader(ds='usps', processed='usps_3_500.pt', batch_num=batch_size, shuffle=True)
    # dataloader = get_usps_loader(batch_size=batch_size, shuffle=True)

    GE_chain = ichain(generator.parameters(), generator.parameters())
    optimizer_GE = torch.optim.Adam(GE_chain, lr=lr1, betas=(b1, b2), weight_decay=decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr1, betas=(b1, b2), weight_decay=decay)

    print('\nBegin training session with %i epochs...\n' % (n_epochs))
    for epoch in tqdm(range(n_epochs)):
        for i, (imgs, labels) in tqdm(enumerate(dataloader)):
            generator.train()
            encoder.train()
            generator.zero_grad()
            discriminator.zero_grad()
            encoder.zero_grad()

            # real_imgs = [imgs.type(Tensor) for _ in range(2)]
            real_imgs = imgs.type(Tensor)

            # ---------------------------
            #  Train Generator + Encoder
            # ---------------------------
            optimizer_GE.zero_grad()
            zn, zc, zc_idx = sample_z(shape=imgs.shape[0], latent_dim=latent_dim, n_c=n_c)
            gen_imgs = generator(zn, zc)
            D_gen = discriminator(gen_imgs)
            D_real = discriminator(real_imgs)
            if (i % n_skip_iter == 0):
                enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encoder(gen_imgs)
                zn_loss = mse_loss(enc_gen_zn, zn)
                zc_loss = xe_loss(enc_gen_zc_logits, zc_idx)
                if wass_metric:
                    ge_loss = torch.mean(D_gen) + betan1 * zn_loss + betac1 * zc_loss
                else:
                    valid = Variable(Tensor(gen_imgs[0].size(0), 1).fill_(1.0), requires_grad=False)
                    v_loss = bce_loss(D_gen[0], valid)
                    ge_loss = v_loss + betan1 * zn_loss + betac1 * zc_loss
                ge_loss.backward(retain_graph=True)
                optimizer_GE.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            if wass_metric:
                grad_penalty = calc_gradient_penalty(discriminator, real_imgs, gen_imgs)
                d_loss = torch.mean(D_real) - torch.mean(D_gen) + grad_penalty
            else:
                fake = Variable(Tensor(gen_imgs[0].size(0), 1).fill_(0.0), requires_grad=False)
                real_loss = bce_loss(D_real[0], valid)
                fake_loss = bce_loss(D_gen[0], fake)
                d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
        # ----
        # each epoch
        # ----
        print("[Epoch {}/{}] \n\tModel Losses: [D1: {}] [GE1: {}]".format(
            epoch, n_epochs, d_loss.item(), ge_loss.item()))
    # ----
    # after epoch
    # ----
    torch.save(generator.state_dict(), os.path.join(models_dir, 'generator.pt'))
    torch.save(discriminator.state_dict(), os.path.join(models_dir, 'discriminator.pt'))
    torch.save(encoder.state_dict(), os.path.join(models_dir, 'encoder.pt'))


if __name__ == "__main__":
    main()
