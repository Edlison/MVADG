import cv2
import sys

import torch

sys.path.append("./img_gist_feature/")

from img_gist_feature.utils_gist import *
import matplotlib.pyplot as plt


tar_f = '../../data/MNIST/processed/training.pt'
imgs, labels = torch.load(tar_f)
img = imgs[0].numpy()
print('before ', img.shape)

gist_helper = GistUtils()
np_gist = gist_helper.get_gist_vec(img, mode="gray")
print("after ", np_gist.shape)


