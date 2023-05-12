# @Author  : Edlison
# @Date    : 4/29/21 11:05
import torch
from v5.Dataset import gen_cluster_set, ZooDataset
from v6.Eval import show
from v6.clusgan.datasets import UspsDataset, UspsAnomDataset, MNISTDataset, MNISTAnomDataset, get_loader, get_anom_loader, get_multi_loader, MultiDataset

# loader = get_anom_dataloader(processed='anom_37_4000_4000.pt', batch_num=25, shuffle=False)
# imgs, labels = next(iter(loader))
# show(imgs, 5, 5)

# loader = get_loader('usps', batch=25, shuffle=True)
# imgs, labels = next(iter(loader))
# show(imgs, 5, 5)
# print(labels)

# loader = get_multi_loader(batch=25, shuffle=True)
#
# (mnist_imgs, mnist_labels), (usps_imgs, usps_labels) = next(iter(loader))
# show(mnist_imgs, 5, 5)
# show(usps_imgs, 5, 5)
# print(usps)
