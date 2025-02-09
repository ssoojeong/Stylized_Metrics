
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.transforms as T
import os
import clip
import torch
# from torchvision.datasets import CIFAR100
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from tqdm import tqdm
import argparse


#vgg style, content
from styleloss.vggloss_gram import PerceptualLoss
from collections import OrderedDict
import warnings

import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision.transforms import ToTensor
import os
import numpy as np


def metric_clip(stylized_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)
    img_path2 = stylized_path
    
    total = []
    for content_num in tqdm(sorted(os.listdir(img_path2))[:200]):
        img_path2_file = os.path.join(img_path2, content_num)
    
        #org-content-1개
        img_path1_file=f'/userHome/userhome1/sojeong/webtoonization/2412/dataset/ffhq/{content_num}'
        
        image1 = Image.open(img_path1_file)
        image2 = Image.open(img_path2_file)
                        
        if len(preprocess(image1).size()) !=4: 
            image_input1 = preprocess(image1).unsqueeze(0).to(device)
            image_input2 = preprocess(image2).unsqueeze(0).to(device)
        else:
            image_input1 = preprocess(image1).to(device)
            image_input2 = preprocess(image2).to(device)
            
        with torch.no_grad():
            image_features1 = model.encode_image(image_input1)
            image_features2 = model.encode_image(image_input2)
            
        # Pick the top 5 most similar labels for the image
        image_features1 /= image_features1.norm(dim=-1, keepdim=True)
        image_features2 /= image_features2.norm(dim=-1, keepdim=True)
        similarity = image_features1 @ image_features2.T
        values, indices = similarity[0].topk(1)
        total.append(values.item())
    
    return -np.log(np.mean(total))
    

def metric_vgg(norm_type, stylized_path, style_images): 
    tf = ToTensor()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_path2 = stylized_path
                 
    total = []     
    for content_num in tqdm(sorted(os.listdir(img_path2))[:200]):
        img_path2_file = os.path.join(os.path.join(img_path2, f'{content_num}'))

        if norm_type == 'content':
            img_path1=f'/userHome/userhome1/sojeong/webtoonization/2412/dataset/ffhq/{content_num}'
            img_path1_file = os.path.join(img_path1)
        
        elif norm_type == 'style':
            img_path1=style_images
            img_path1_file = os.path.join(img_path1, os.listdir(img_path1)[0])
        

        img_1 = tf(Image.open(img_path1_file)).to(device)
        img_2 = tf(Image.open(img_path2_file)).to(device)
        
        ###추가###
        if len(img_1.size())==3: img_1 = img_1.unsqueeze(0)
        if len(img_2.size())==3: img_2 = img_2.unsqueeze(0)
        
        if norm_type == 'style':
            total.append(np.sum([i.cpu() for i in PerceptualLoss(norm_type=f'{norm_type}', device=device)(img_1, img_2)]))
        elif norm_type == 'content':
            total.append(np.mean([i.cpu() for i in PerceptualLoss(norm_type=f'{norm_type}', device=device)(img_1, img_2)]))
        else:
            break

    return np.mean(total)
