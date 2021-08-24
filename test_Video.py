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
    parser.add_argument('--netGDN', default='', help="path to netGDN")
    parser.add_argument('--netEdge', default='', help="path to netEdge")
    parser.add_argument('--netMerge', default='', help="path to netMerge")
    parser.add_argument('--video_path', type=str, default='', help='path of the input video')
    parser.add_argument('--save_path', type=str, default='', help='path to save the ouput video')
    parser.add_argument('--imageSize_h', type=int,
      default=512, help='the height of the cropped input image to network')
    parser.add_argument('--imageSize_w', type=int,
      default=512, help='the width of the cropped input image to network')

    opt = parser.parse_args()
    print(opt)

    opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)
    print("Random Seed: ", opt.manualSeed)


    device = torch.device("cuda:0")


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
    net_metric = net_metric.to(device)
    utils.set_requires_grad(net_metric, requires_grad=False)


    stream = cv2.VideoCapture(opt.video_path)
    FPS = stream.get(cv2.CAP_PROP_FPS)
    codec = cv2.VideoWriter_fourcc('D','I','V','X')
    video_out = cv2.VideoWriter(opt.save_path, codec, FPS, (opt.imageSize_w, opt.imageSize_h), 1)

    mean=(0.5, 0.5, 0.5)
    std=(0.5, 0.5, 0.5)
    import transforms.pix2pix as transforms
    transform1=transforms.Compose1([
      transforms.Scale1(opt.imageSize_h, opt.imageSize_w),
      transforms.CenterCrop1(opt.imageSize_h, opt.imageSize_w),
    ])
    transform2=transforms.Compose1([
      transforms.ToTensor1(),
      transforms.Normalize1(mean, std),
    ])
    transform3=transforms.Compose1([
      transforms.Scale1(384, 384),
      transforms.ToTensor1(),
      transforms.Normalize1(mean, std),
    ])

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

    time_start=time.time()
    while True:
        ret, frame = stream.read()
        if not ret:
            print("unable to fetch frame")
            break
        #from PIL import Image
        #img = Image.open(path).convert('RGB')

        #frame = frame.transpose((2, 0, 1))
        #frame = frame[np.newaxis,:,:,:]
        input = Image.fromarray(frame)
        input = transform1(input)
        down_input = transform3(input)
        input = transform2(input)

        input = input.unsqueeze_(0)
        down_input = down_input.unsqueeze_(0)

        input = input.to(device)
        down_input = down_input.to(device)

        i_G_x = conv1(down_input)
        i_G_y = conv2(down_input)
        input_edge = torch.tanh(torch.abs(i_G_x)+torch.abs(i_G_y))

        pred_edge = netEdge(torch.cat([down_input, input_edge], 1))
        demoire_down = netGDN(down_input, pred_edge)
        x_hat = netMerge(demoire_down, input)

        ti1 = x_hat[0,:,:,:]
        #mi1 = cv2.cvtColor(utils.my_tensor2im(ti1), cv2.COLOR_BGR2RGB)
        mi1 = utils.my_tensor2im(ti1)
        cv2.imwrite("current_frame.png", mi1)
        video_out.write(mi1)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
          break
    time_end=time.time()
    print('Time spent: ',time_end - time_start)
    stream.release()
    video_out.release()