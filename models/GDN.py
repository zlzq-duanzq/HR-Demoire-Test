import torch
import torch.nn as nn
from models.DenseBlock import *
from models.NEDB import NEDB

class GDN(nn.Module):
  def __init__(self):
    super(GDN, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
    self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
    self.conv1_e = nn.Conv2d(3, 64, 3, 1, 1)
    self.se_tmp = SEBlock(128, 64)


    self.dense_block1=BottleneckBlock2(64,64)
    self.trans_block1=TransitionBlock1(128,64)

    self.dense_block2=BottleneckBlock2(64,64)
    self.trans_block2=TransitionBlock1(128,64)

    self.dense_block3=BottleneckBlock2(64,64)
    self.trans_block3=TransitionBlock1(128,64)

    self.nb2 = NEDB(block_num=4, inter_channel=32, channel=64)

    self.dense_block4=BottleneckBlock2(64,64)
    self.trans_block4=TransitionBlock(128,64)

    self.dense_block5=BottleneckBlock2(128,64)
    self.trans_block5=TransitionBlock(192,64)

    self.dense_block6=BottleneckBlock2(128,64)
    self.trans_block6=TransitionBlock(192,64)
    self.se5 = SEBlock(448, 220)

    self.tanh = nn.Tanh()
    self.relu = nn.LeakyReLU(0.2, inplace=True)
    self.upsample = F.upsample_nearest

    self.conv11 = nn.Conv2d(64, 64, kernel_size=1,stride=1,padding=0) 
    self.conv21 = nn.Conv2d(64, 64, kernel_size=1,stride=1,padding=0) 
    self.conv31 = nn.Conv2d(64, 64, kernel_size=1,stride=1,padding=0) 
    self.conv41 = nn.Conv2d(128, 64, kernel_size=1,stride=1,padding=0)
    self.conv51 = nn.Conv2d(128, 64, kernel_size=1,stride=1,padding=0)
    self.conv61 = nn.Conv2d(64, 64, kernel_size=1,stride=1,padding=0) 

    self.fusion = nn.Sequential(
        nn.Conv2d(448, 64, 1, 1, 0),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 64, 3, 1, 1),
        nn.LeakyReLU(0.2, inplace=True)
    )


    self.final_conv = nn.Sequential(
      nn.Conv2d(64, 3, 3, 1, 1),
      nn.Tanh(),
    )

  def forward(self, x, e):

    f0 = self.conv1(x)
    f0_e = self.conv1_e(e)
    tmp = torch.cat([f0, f0_e], 1)
    tmp = self.se_tmp(tmp)
    f1 = self.conv2(tmp)

    old_state1, old_state2 = None, None
    x1 = self.dense_block1(f1)
    x1 = self.trans_block1(x1)
    x2 = (self.dense_block2(x1))
    x2 = self.trans_block2(x2)
    x3 = (self.dense_block3(x2))
    x3 = self.trans_block3(x3)

    x3, st1, st2 = self.nb2(x3, old_state1, old_state2)
    # x3, st = self.gru(x3, old_state)
    x4 = (self.dense_block4(x3))
    x4 = self.trans_block4(x4)
    x4 = torch.cat([x4, x2], 1)
    x5 = (self.dense_block5(x4))
    x5 = self.trans_block5(x5)
    x5 = torch.cat([x5, x1], 1)
    x6 = (self.dense_block6(x5))
    x6 = (self.trans_block6(x6))

    shape_out = x6.data.size()
    shape_out = shape_out[2:4]
    x11 = self.upsample(self.relu((self.conv11(x1))), size=shape_out)
    x21 = self.upsample(self.relu((self.conv21(x2))), size=shape_out)
    x31 = self.upsample(self.relu((self.conv31(x3))), size=shape_out)
    x41 = self.upsample(self.relu((self.conv41(x4))), size=shape_out)
    x51 = self.upsample(self.relu((self.conv51(x5))), size=shape_out)
    x61 = self.relu((self.conv61(x6)))
    r1 = torch.cat([f1, x11, x21, x31, x41, x51, x61], dim=1)
    r1 = self.se5(r1)
    r1 = self.fusion(r1)
    x = self.final_conv(f0 + r1)
    return x
