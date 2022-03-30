import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from scipy.misc import imread, imresize
from PIL import Image
import os

image_path = "/home/zhouc/tyb/caption/video_captioning_rl/graph/frame/"
images = os.listdir(image_path)
smoth = 'True'
alphas = [9.7720e-01, 1.4931e-02, 5.1484e-03, 1.2387e-03, 4.3621e-04, 1.5376e-04,
        1.1025e-04, 1.0718e-04, 1.1664e-04, 9.6557e-05, 5.5635e-05, 4.7762e-05,
        6.1694e-05, 1.0084e-04, 4.1810e-05, 5.0099e-05, 2.4396e-05, 2.0682e-05,
        1.2282e-05, 1.2458e-05, 1.2412e-05, 4.2889e-06, 2.1919e-06, 1.2610e-06,
        2.4615e-06, 4.3483e-06, 4.2459e-06, 3.2303e-06]
for i in images:
    image = Image.open(image_path+i)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = ['a' 'girl' 'is' 'walking' 'on' 'the' 'road']



    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t]
        if 1:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha, [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()