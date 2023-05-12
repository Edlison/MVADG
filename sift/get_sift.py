# @Author  : Edlison
# @Date    : 5/30/21 23:26

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from v6.clusgan.datasets import get_loader

# Load
imgname = 'Image.jpeg'
tar_f = '../../data/MNIST/processed/training.pt'
imgs, labels = torch.load(tar_f)

# Calculate
sift = cv2.SIFT_create()
# img = cv2.imread(imgname)
img = imgs[7]
img.unsqueeze_(-1)
img = img.numpy()
# gray = cv2.cvtColor(img.numpy(), cv2.COLOR_BGR2GRAY)
kp, des = sift.detectAndCompute(img, None)

plt.imshow(img)
plt.title('before')
plt.show()
print(img.shape)

# Draw sift
# img1 = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
mat = np.zeros([28, 28, 3])
img1 = cv2.drawKeypoints(img, kp, mat, color=(255, 0, 0))

plt.imshow(img1)
plt.title('after')
plt.show()
print(img1.shape)

plt.imshow(mat)
plt.title('mat')
plt.show()