from collections import OrderedDict
import warnings

import torch
import torch.nn as nn
import torchvision.models.vgg as vgg
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision.transforms import ToTensor
import os

tf = ToTensor()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

__all__ = ['VGG']

NAMES = {
    'vgg11': [
        'conv1_1', 'relu1_1', 'pool1',
        'conv2_1', 'relu2_1', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5',
    ],
    'vgg13': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'pool5',
    ],
    'vgg16': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1',
        'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
        'conv4_1', 'relu4_1',
        'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
        'conv5_1', 'relu5_1',
        'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5',
    ],
    'vgg19': [
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
        'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
        'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
        'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
    ]
}


def insert_bn(names: list):
    """
    Inserts bn layer after each conv.

    Parameters
    ---
    names : list
        The list of layer names.
    """
    names_bn = []
    for name in names:
        names_bn.append(name)
        if 'conv' in name:
            pos = name.replace('conv', '')
            names_bn.append('bn' + pos)
    return names_bn


class VGG(nn.Module):
    """
    Creates any type of VGG models.

    Parameters
    ---
    model_type : str
        The model type you want to load.
    requires_grad : bool, optional
        Whethere compute gradients.
    """
    def __init__(self, model_type: str, requires_grad: bool = False):
        super(VGG, self).__init__()

        features = getattr(vgg, model_type)(True).features
        self.names = NAMES[model_type.replace('_bn', '')]
        if 'bn' in model_type:
            self.names = insert_bn(self.names)

        self.net = nn.Sequential(OrderedDict([
            (k, v) for k, v in zip(self.names, features)
        ]))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.register_buffer(
            name='vgg_mean',
            tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]],
                                requires_grad=False)
        )
        self.register_buffer(
            name='vgg_std',
            tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]],
                                requires_grad=False)
        )

    def z_score(self, x):
        x = x.sub(self.vgg_mean.detach())
        x = x.div(self.vgg_std.detach())
        return x

    def forward(self, x: torch.Tensor, targets: list) -> dict:
        """
        Parameters
        ---
        x : torch.Tensor
            The input tensor normalized to [0, 1].
        target : list of str
            The layer names you want to pick up.
        Returns
        ---
        out_dict : dict of torch.Tensor
            The dictionary of tensors you specified.
            The elements are ordered by the original VGG order. 
        """

        assert all([t in self.names for t in targets]),\
            'Specified name does not exist.'

        if torch.all(x < 0.) and torch.all(x > 1.):
            warnings.warn('input tensor is not normalize to [0, 1].')

        x = self.z_score(x)

        out_dict = OrderedDict()
        for key, layer in self.net._modules.items():
            x = layer(x)
            if key in targets:
                out_dict.update({key: x})
            if len(out_dict) == len(targets):  # to reduce wasting computation
                break

        return out_dict
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

#from models import VGG

######0501#######
def gram_matrix(input):
    a, b, c, d = input.size()  # a=배치 크기(=1)
    # b=특징 맵의 수
    # (c,d)=특징 맵의 차원 (N=c*d)

    features = input.view(a * b, c * d)  # F_XL을 \hat F_XL로 크기 조정

    G = torch.mm(features, features.t())  # gram product를 계산

    # 각 특징 맵이 갖는 값의 수로 나누어
    # gram 행렬의 값을 '정규화'
    return G.div(a * b * c * d)
##################

class PerceptualLoss(nn.Module):
    """
    PyTorch module for perceptual loss.

    Parameters
    ---
    model_type : str
        select from [`vgg11`, `vgg11bn`, `vgg13`, `vgg13bn`,
                     `vgg16`, `vgg16bn`, `vgg19`, `vgg19bn`, ].
    target_layers : str
        the layer name you want to compare.
    norm_type : str
        the type of norm, select from ['mse', 'fro']
    """
    def __init__(self,
                model_type: str = 'vgg19',
                #target_layer: str = 'relu5_1',
                #norm_type: str = 'mse'):
                norm_type: str = 'style',
                device=device):
        super(PerceptualLoss, self).__init__()

        #assert norm_type in ['mse', 'fro']
        assert norm_type in ['mse', 'style', 'content']

        model_type = 'vgg19'
        
        self.model = VGG(model_type=model_type)
        self.target_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        #self.target_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
        self.content_target_layers= ['conv4_1']
        self.norm_type = norm_type
        #self.transform = torch.nn.functional.interpolate
        self.transform = T.Resize(size = (224,224))
        self.device=device

    def forward(self, x, y):
        # with torch.no_grad():
        #loss = 0.0
        loss = []
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        x, y =  self.transform(x), self.transform(y)
        for target_layer in self.target_layers:
            x_feat, *_ = self.model(x, [target_layer]).values()
            y_feat, *_ = self.model(y, [target_layer]).values()
            # print('### x_feat size:', x_feat.size())
            # print('### y_feat size:', y_feat.size())
            # frobenius norm in the paper, but mse loss is actually used in
            # https://github.com/ZZUTK/SRNTT/blob/master/SRNTT/model.py#L376.
            
            #1.original
            # if self.norm_type == 'mse':
            #     loss += F.mse_loss(x_feat, y_feat, reduction='mean')
            # elif self.norm_type == 'fro':
            #     loss += torch.norm(x_feat - y_feat, p='fro')
    
        # return loss/len(self.target_layers)
    
            #2.변형
            if self.norm_type == 'mse':
                loss.append(F.mse_loss(x_feat, y_feat, reduction='mean'))
            elif self.norm_type == 'style':
                target = gram_matrix(x_feat).detach()
                G = gram_matrix(y_feat)
                loss.append(F.mse_loss(target, G, reduction='mean'))
            elif self.norm_type == 'content' and target_layer in self.content_target_layers:
                loss.append(F.mse_loss(x_feat, y_feat, reduction='mean'))

            # elif self.norm_type == 'fro':
            #     loss.append(torch.norm(x_feat - y_feat, p='fro'))

        return loss


warnings.filterwarnings(action='ignore')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
        "--save_dir",
        type=str,
        default='/userHome/userhome1/sojeong/2401',
        #default='/userHome/userhome1/sojeong/2401/0222_metric_3/coco/1_nh/0.2',
        
        
        help="path to save results"
    )
parser.add_argument(
        "--ref_dir",
        type=str,
        #default="/userHome/userhome1/sojeong/2401/data/wikiart/1_h",
        default="/userHome/userhome1/sojeong/2401/data/wikiart/1_h",
        help="path to style image_dir"
    )
parser.add_argument(
        "--gen_dir",
        type=str,
        default='/userHome/userhome1/sojeong/2401/VCT_contra/outputs/test_contra_blank/0',
        #default='/userHome/userhome1/sojeong/2401/0222_data/3_vct/1_nh/0.2',
        
        help="path to gen image_dir"
    )
parser.add_argument('-c', '--gpu', default='', type=int,
                    help='GPU to use (leave blank for CPU only)')


def main(args):

    os.makedirs(args.save_dir, exist_ok=True) ##추가
    
    #gpu
    torch.cuda.set_device(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #exec
    ref_dir = sorted(os.listdir(args.ref_dir))
    gen_dir = sorted([k for k in os.listdir(args.gen_dir) if os.path.isfile(os.path.join(args.gen_dir, k))]) #gen_dir: 특정 class_특정 ref 개수 #파일만 추출
    #class_id = args.ref_dir.split('/')[-2]
    ref_count = len(ref_dir)
    gen_count = len(gen_dir)
    total_avg = []
    
    #resume
    exist_file_path = os.path.join(args.save_dir, 'gramloss.txt')
    if os.path.exists(exist_file_path):
        with open(exist_file_path, 'r') as f:
            #print(f.readlines()[-1].split(':')[0])
            total = f.readlines()
            
            idx = len(total)
            gen_dir = gen_dir[idx:]
            gen_count = len(gen_dir)
            #raw_dir = raw_dir[idx:]
            
            #assert len(gen_dir)==len(raw_dir) and gen_dir[0].split('.')[0]==raw_dir[0].split('.')[0]
            for i in total:
                value = float(i.split(':')[1][1:]) #gramloss value
                total_avg.append(value)
    
    #pbar 수동 업데이트
    with tqdm(total=len(gen_dir)) as pbar:
        for i, gen_list in enumerate(gen_dir):
            id = gen_list.split('.')[0]
            loss = 0.0
            avg = 0.0
            gen_img = tf(Image.open(os.path.join(args.gen_dir, gen_list))).to(device)
            for ref_list in ref_dir:
                ref_img = tf(Image.open(os.path.join(args.ref_dir, ref_list))).to(device)
                ###추가###
                if len(ref_img.size())==3: ref_img = ref_img.unsqueeze(0)
                if len(gen_img.size())==3: gen_img = gen_img.unsqueeze(0)
                #########
                loss += PerceptualLoss()(ref_img, gen_img)
            avg = float(loss/ref_count)
            total_avg.append(avg)
            with open(os.path.join(args.save_dir, 'gramloss.txt'), 'a') as f:
            #with open(os.path.join(args.save_dir, f'{class_id}_{ref_count}.txt'), 'a') as f:
            #with open(os.path.join(args.save_dir, 'New_realism_2.txt'), 'a') as f:
                f.write(f'{id}: '+str(avg))
                f.write('\n')
            #print(f'{i}_avg: {np.mean(total_avg)}, {i}_std: {np.std(total_avg)}')
            pbar.update(1)
        
    print_avg = np.mean(total_avg)
    print_std = np.std(total_avg)
    # print('###########')
    # #print('total_avg: ', total_avg)
    # print('avg: ', print_avg)
    # print('std: ', print_std)
    # print('###########')
    with open(os.path.join(args.save_dir, 'gramloss.txt'), 'a') as f:
        f.write(f'Total_avg: '+str(print_avg))
        f.write('\n')
        f.write(f'Total_std: '+str(print_std))
        f.write('\n')
         

if __name__ == '__main__':
    args = parser.parse_args()
    
    main(args)