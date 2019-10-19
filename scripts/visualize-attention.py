#!/usr/bin/env python
import argparse
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from nmtpytorch.translator import Translator
from nmtpytorch.utils.data import make_dataloader

# visualize_att
from PIL import Image
import matplotlib.pyplot as plt
import skimage.transform
import matplotlib.cm as cm

import os

def visualize_att(image_path, words, alphas, out_path, smooth=True, max_len=100):
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    alphas.resize(alphas.shape[0], 14, 14)

    plt.clf()

    for t in range(len(words)):
        if t > max_len:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]

        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha, upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha, [14 * 24, 14 * 24])

        plt.imshow(alpha, alpha=0.8)

        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.savefig(out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='visualize-attention',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="generate visualized attention over the image",
        argument_default=argparse.SUPPRESS)

    parser.add_argument('-a', '--attention', type=str, help='pkl file storing attentions')
    parser.add_argument('-s', '--image_split', type=str, help='image-split file name')
    parser.add_argument('-i', '--images', type=str, help='images directory')
    parser.add_argument('-o', '--output', type=str, help='image output directory.')

    args = parser.parse_args()
    assert os.path.exists(args.image_split)
    assert os.path.exists(args.images)

    if os.path.exists(args.output) == False:
        os.mkdir(args.output)

    with open(args.attention, 'br') as f:
        data = pkl.load(f)
    
    for idx, (d, img_name) in tqdm(enumerate(zip(data, open(args.image_split)))):
        if not ('sec_att' in d):
            continue

        img_path = f'{args.images}/{img_name.strip()}'
        out_path = f'{args.output}/{idx}'

        visualize_att(img_path, d['hyp'].split(), d['sec_att'], out_path)
