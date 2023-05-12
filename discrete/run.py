# @Author  : Edlison
# @Date    : 5/11/21 20:50
from v6.discrete.models import Generator, Discriminator, Encoder
from v6.discrete.utils import ZooDataset, sample_z, save_models, WineDataset
from v6.discrete.test import eval_wine_epoch, eval_zoo_epoch, show_loss_auc, eval_encoder_multi, eval_encoder
import torch
from torch.utils.data import DataLoader
from itertools import chain
from torch import optim


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def run(latent_dim, cls_num, features_dim, beta_zn, beta_zc, lr, epochs, batch_size, shuffle):
    loader_zoo = DataLoader(ZooDataset(), batch_size=batch_size, shuffle=shuffle)
    loader_wine = DataLoader(WineDataset(anom=True), batch_size=batch_size, shuffle=shuffle)

    generator = Generator(latent_dim, cls_num, features_dim=features_dim, discrete=True)
    discriminator = Discriminator(features_dim=features_dim)
    encoder = Encoder(latent_dim, cls_num, features_dim=features_dim)

    GE_chain = chain(generator.parameters(), encoder.parameters())
    GE_optim = optim.RMSprop(GE_chain, lr=lr)
    D_optim = optim.RMSprop(discriminator.parameters(), lr=lr)

    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    weight_clip = 0.01

    if cuda:
        generator.cuda()
        discriminator.cuda()
        encoder.cuda()

    for epoch in range(epochs):
        for i, (data, labels) in enumerate(loader_zoo):
            generator.train()
            discriminator.train()
            encoder.train()

            data = data.type(Tensor)

            zn, zc, zc_idx = sample_z(data.shape[0], latent_dim=latent_dim, n_c=cls_num)

            gen_data = generator(zn, zc)
            d_fake = discriminator(gen_data)
            d_real = discriminator(data)

            enc_zn, enc_zc, enc_zc_logits = encoder(gen_data)

            # Train G+E
            zn_loss = mse_loss(enc_zn, zn)
            zc_loss = xe_loss(enc_zc_logits, zc_idx)
            ge_loss = -torch.mean(d_fake) + beta_zn * zn_loss + beta_zc * zc_loss

            GE_optim.zero_grad()
            ge_loss.backward(retain_graph=True)
            GE_optim.step()

            # Train D
            d_loss = -(torch.mean(d_real) - torch.mean(d_fake))

            D_optim.zero_grad()
            d_loss.backward()
            D_optim.step()

            for p in discriminator.parameters():
                p.data.clamp_(-weight_clip, weight_clip)

        print('[Epoch {}/{}]\nGE_loss: {}, D_loss: {}.'.format(epoch, epochs, ge_loss, d_loss))
        print('GE_loss: {} | d_fake: {} | zn_loss: {} | zc_loss: {}.'.format(ge_loss, -torch.mean(d_fake), zn_loss, zc_loss))
    save_models(generator, discriminator, encoder)


def run_multi(latent_dim, cls_num, view1_features_dim, view2_features_dim, beta_zn, beta_zc, lr, epochs, batch_size, shuffle, discrete=False):
    # loader_wine = DataLoader(WineDataset(anom=True), batch_size=batch_size, shuffle=shuffle)
    loader_zoo = DataLoader(ZooDataset(anom=True), batch_size=batch_size, shuffle=shuffle)

    generator1 = Generator(latent_dim, cls_num, features_dim=view1_features_dim, discrete=discrete)
    generator1.name = 'Generator1'
    generator2 = Generator(latent_dim, cls_num, features_dim=view2_features_dim, discrete=discrete)
    generator2.name = 'Generator2'
    discriminator1 = Discriminator(features_dim=view1_features_dim, discrete=discrete)
    discriminator1.name = 'Discriminator1'
    discriminator2 = Discriminator(features_dim=view2_features_dim, discrete=discrete)
    discriminator2.name = 'Discriminator2'
    encoder1 = Encoder(latent_dim, cls_num, features_dim=view1_features_dim, discrete=discrete)
    encoder1.name = 'Encoder1'
    encoder2 = Encoder(latent_dim, cls_num, features_dim=view2_features_dim, discrete=discrete)
    encoder2.name = 'Encoder2'

    GE_chain1 = chain(generator1.parameters(), encoder1.parameters())
    GE_chain2 = chain(generator2.parameters(), encoder2.parameters())
    GE_optim1 = optim.RMSprop(GE_chain1, lr=lr)
    GE_optim2 = optim.RMSprop(GE_chain2, lr=lr)
    D_optim1 = optim.RMSprop(discriminator1.parameters(), lr=lr)
    D_optim2 = optim.RMSprop(discriminator2.parameters(), lr=lr)

    xe_loss1 = torch.nn.CrossEntropyLoss()
    xe_loss2 = torch.nn.CrossEntropyLoss()
    mse_loss1 = torch.nn.MSELoss()
    mse_loss2 = torch.nn.MSELoss()

    weight_clip = 0.01

    if cuda:
        generator1.cuda()
        generator2.cuda()
        discriminator1.cuda()
        discriminator2.cuda()
        encoder1.cuda()
        encoder2.cuda()

    eval_ge_loss1 = []
    eval_ge_loss2 = []
    eval_d_loss1 = []
    eval_d_loss2 = []
    eval_auc = []

    for epoch in range(epochs):
        for i, (data1, data2, labels) in enumerate(loader_zoo):
            generator1.train()
            generator2.train()
            discriminator1.train()
            discriminator2.train()
            encoder1.train()
            encoder2.train()

            data1 = data1.type(Tensor)
            data2 = data2.type(Tensor)

            zn, zc, zc_idx = sample_z(data1.shape[0], latent_dim=latent_dim, n_c=cls_num)

            gen_data1 = generator1(zn, zc)
            gen_data2 = generator2(zn, zc)
            d_fake1 = discriminator1(gen_data1)
            d_fake2 = discriminator2(gen_data2)
            d_real1 = discriminator1(data1)
            d_real2 = discriminator2(data2)

            enc_zn1, enc_zc1, enc_zc_logits1 = encoder1(gen_data1)
            enc_zn2, enc_zc2, enc_zc_logits2 = encoder2(gen_data2)

            # Train G+E
            zn_loss1 = mse_loss1(enc_zn1, zn)
            zn_loss2 = mse_loss2(enc_zn2, zn)
            zc_loss1 = xe_loss1(enc_zc_logits1, zc_idx)
            zc_loss2 = xe_loss2(enc_zc_logits2, zc_idx)
            ge_loss1 = -torch.mean(d_fake1) + beta_zn * zn_loss1 + beta_zc * zc_loss1
            ge_loss2 = -torch.mean(d_fake2) + beta_zn * zn_loss2 + beta_zc * zc_loss2

            GE_optim1.zero_grad()
            ge_loss1.backward(retain_graph=True)
            GE_optim1.step()
            GE_optim2.zero_grad()
            ge_loss2.backward(retain_graph=True)
            GE_optim2.step()

            # Train D
            d_loss1 = -(torch.mean(d_real1) - torch.mean(d_fake1))
            d_loss2 = -(torch.mean(d_real2) - torch.mean(d_fake2))

            D_optim1.zero_grad()
            d_loss1.backward()
            D_optim1.step()
            D_optim2.zero_grad()
            d_loss2.backward()
            D_optim2.step()

            for p1, p2 in zip(discriminator1.parameters(), discriminator2.parameters()):
                p1.data.clamp_(-weight_clip, weight_clip)
                p2.data.clamp_(-weight_clip, weight_clip)

        print('[Epoch {}/{}]\nGE_loss1: {} | GE_loss2: {} | D_loss1: {} | D_loss2: {}'.format(epoch, epochs, ge_loss1, ge_loss2, d_loss1, d_loss2))
        # auc = eval_wine_epoch(encoder1, encoder2)
        auc = eval_zoo_epoch(encoder1, encoder2)
        eval_ge_loss1.append(ge_loss1)
        eval_ge_loss2.append(ge_loss2)
        eval_d_loss1.append(d_loss1)
        eval_d_loss2.append(d_loss2)
        eval_auc.append(auc)

    show_loss_auc(eval_ge_loss1, eval_ge_loss2, eval_d_loss1, eval_d_loss2, eval_auc)
    save_models(generator1, discriminator1, encoder1)
    save_models(generator2, discriminator2, encoder2)


if __name__ == '__main__':
    epochs = 300
    latent_dim = 30
    cls_num = 7
    beta_zn = 1
    beta_zc = 1
    lr = 0.001
    view1_features_dim = 8
    view2_features_dim = 7
    batch_size = 9
    shuffle = True
    discrete = True

    # run(latent_dim, cls_num, features_dim, beta_zn, beta_zc, lr, epochs, batch_size, shuffle)
    run_multi(latent_dim, cls_num, view1_features_dim, view2_features_dim, beta_zn, beta_zc, lr, epochs, batch_size, shuffle, discrete)

    # eval_encoder_multi(batch=135)
    # eval_encoder(batch=178)
