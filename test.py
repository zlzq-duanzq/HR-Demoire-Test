from __future__ import print_function
import argparse
import os
import sys
import random
import time
import pdb
from PIL import Image
import math
import numpy as np
import cv2
#from skimage.measure import compare_ssim as ssim
from skimage.metrics import structural_similarity as ssim
#from skimage.measure import compare_psnr as Psnr
from skimage.metrics import peak_signal_noise_ratio as Psnr
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
cudnn.benchmark = True
cudnn.fastest = True

from misc import *
from myutils import utils
import models_metric
import time

import models.GDN as net
import models.MergeNet as merge
import models.EdgePredict as edge

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=False,
      default='my_loader',  help='name for the dataset loader')
    parser.add_argument('--dataroot', required=False,
      default='', help='path to trn dataset')
    parser.add_argument('--netGDN', default='', help="path to netGDN")
    parser.add_argument('--netEdge', default='', help="path to netEdge")
    parser.add_argument('--netMerge', default='', help="path to netMerge")
    parser.add_argument('--kernel_size', type=int, default=8, help='patch size for dct')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--originalSize_h', type=int,
      default=539, help='the height of the original input image')
    parser.add_argument('--originalSize_w', type=int,
      default=959, help='the height of the original input image')
    parser.add_argument('--imageSize_h', type=int,
      default=512, help='the height of the cropped input image to network')
    parser.add_argument('--imageSize_w', type=int,
      default=512, help='the width of the cropped input image to network')
    parser.add_argument('--pre', type=str, default='', help='prefix of different dataset')
    parser.add_argument('--image_path', type=str, default='', help='path to save the evaluated image')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    parser.add_argument('--record', type=str, default='default.txt', help='file to record scores for each image')
    parser.add_argument('--write', type=int, default=0, help='determine whether we save the result images')
    opt = parser.parse_args()
    print(opt)

    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    print("Random Seed: ", opt.manualSeed)

    val_dataloader = getLoader(opt.dataset,
                          opt.dataroot,
                          opt.originalSize_h,
                          opt.originalSize_w,
                          opt.imageSize_h,
                          opt.imageSize_w,
                          opt.batchSize,
                          opt.workers,
                          mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
                          split='test',
                          shuffle=False,
                          seed=opt.manualSeed,
                          pre=opt.pre)


    if opt.write==0:
        print('no')
    else:
        print('yes')

    device = torch.device("cuda:1")

    # dfine and load models 
    netGDN = net.GDN()
    if opt.netGDN != '':
        print("load pre-trained GDN model!!!!!!!!!!!!!!!!!")
        netGDN.load_state_dict(torch.load(opt.netGDN))
    netGDN.eval()
    utils.set_requires_grad(netGDN, False)
    netGDN.to(device)


    netEdge = edge.EdgePredict()
    if opt.netEdge != '':
        print("load pre-trained netEdge model!!!!!!!!!!!!!!!!!")
        netEdge.load_state_dict(torch.load(opt.netEdge))
    netEdge.eval()
    utils.set_requires_grad(netEdge, False)
    netEdge.to(device)


    netMerge = merge.MergeNet()
    if opt.netMerge != '':
        print("load pre-trained MergeNet model!!!!!!!!!!!!!!!!!")
        netMerge.load_state_dict(torch.load(opt.netMerge))
    netMerge.eval()
    utils.set_requires_grad(netMerge, False)
    netMerge.to(device)
    
    # load metric
    net_metric = models_metric.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, spatial=False)
    net_metric = net_metric.cuda()
    utils.set_requires_grad(net_metric, requires_grad=False)

    my_psnr = 0
    my_ssim_multi = 0
    patch_size = 384
    res = 0
    cnt1 = 0
    my_time = 0
    f = open(opt.record, "w")
    if os.path.exists(opt.image_path) == False:
        os.makedirs(opt.image_path)
    
    # Sobel kernel conv with 2 directions
    a = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]], dtype=np.float32)
    a = a.reshape(1, 1, 3, 3) # out_c/3, in_c, w, h
    a = np.repeat(a, 3, axis=0)
    conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
    conv1.weight.data.copy_(torch.from_numpy(a))
    conv1.weight.requires_grad = False
    conv1.to(device)

    b = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]], dtype=np.float32)
    b = b.reshape(1, 1, 3, 3)
    b = np.repeat(b, 3, axis=0)
    conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False, groups=3)
    conv2.weight.data.copy_(torch.from_numpy(b))
    conv2.weight.requires_grad = False
    conv2.to(device)
    
    for i, data in enumerate(val_dataloader, 0):
        time_start=time.time()
        # netG.eval()
        print(50*'-')
        print(i)

        input, target, down_input, name = data

        batch_size = input.size(0)
        input = input.to(device)
        target = target.to(device)
        down_input = down_input.to(device)

        i_G_x = conv1(down_input)
        i_G_y = conv2(down_input)
        input_edge = torch.tanh(torch.abs(i_G_x)+torch.abs(i_G_y))

        pred_edge = netEdge(torch.cat([down_input, input_edge], 1))
        demoire_down = netGDN(down_input, pred_edge)
        x_hat = netMerge(demoire_down, input) 

        time_end=time.time()
        # calculate scores
        cnt1+=batch_size
        tmp = torch.sum(net_metric(target, x_hat).detach())
        res += tmp
        L = str(tmp)
        print(res / cnt1)

        my_time += time_end - time_start
        print(f"time_end - time_start: {time_end - time_start}")
        #print(f"total time: {my_time}")
        #print(f"cnt1: {cnt1}")
        print(f"time: {my_time / cnt1}")

        for j in range(x_hat.shape[0]):
            b, c, w, h = x_hat.shape
            ti1 = x_hat[j, :,:,: ]
            tt1 = target[j, :,:,: ]
            mi1 = cv2.cvtColor(utils.my_tensor2im(ti1), cv2.COLOR_BGR2RGB)
            mt1 = cv2.cvtColor(utils.my_tensor2im(tt1), cv2.COLOR_BGR2RGB)
            tmp2 =  Psnr(mt1, mi1)
            my_psnr += tmp2
            tmp3 = ssim(mt1, mi1, multichannel=True)
            my_ssim_multi += tmp3
            L = L +' ' + str(tmp2) +str(tmp3) + '\n'
            f.write(L)
            if opt.write == 1:
                print(f"saving {name[j]}...")
                cv2.imwrite(opt.image_path +os.sep+'res_' + name[j] +'.png', mi1)
        print(my_psnr / cnt1)
        print(my_ssim_multi / cnt1)

    print("avergaed results:")
    print(res / cnt1)
    print(my_psnr / cnt1)
    print(my_ssim_multi / cnt1)
    print(f"time: {my_time / cnt1}")
    f.close()
