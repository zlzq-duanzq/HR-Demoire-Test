import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
  def __init__(self, input_dim, reduction):
    super(SEBlock, self).__init__()
    mid = int(input_dim / reduction)
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.fc = nn.Sequential(
      nn.Linear(input_dim, reduction),
      nn.ReLU(inplace=True),
      nn.Linear(reduction, input_dim),
      nn.Sigmoid()
    )

  def forward(self, x):
    b, c, _, _ = x.size()
    y = self.avg_pool(x).view(b, c)
    y = self.fc(y).view(b, c, 1, 1)
    return x * y

    
class BottleneckBlock(nn.Module):
  def __init__(self, in_planes, out_planes, dropRate=0.0):
    super(BottleneckBlock, self).__init__()
    inter_planes = out_planes * 4
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,padding=0, bias=False)
    self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,padding=1, bias=False)
    self.droprate = dropRate
  def forward(self, x):
    out = self.conv1(self.relu(x))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    out = self.conv2(self.relu(out))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    return torch.cat([x, out], 1)



class BottleneckBlock1(nn.Module):
  def __init__(self, in_planes, out_planes, dropRate=0.0):
    super(BottleneckBlock1, self).__init__()
    inter_planes = out_planes * 4
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,padding=0, bias=False)
    self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=5, stride=1,padding=2, bias=False)
    self.droprate = dropRate
  def forward(self, x):
    out = self.conv1(self.relu(x))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    out = self.conv2(self.relu(out))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    return torch.cat([x, out], 1)


class BottleneckBlock2(nn.Module):
  def __init__(self, in_planes, out_planes, dropRate=0.0):
    super(BottleneckBlock2, self).__init__()
    inter_planes = out_planes * 4
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,padding=0, bias=False)
    self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=7, stride=1,padding=3, bias=False)
    self.droprate = dropRate
  def forward(self, x):
    out = self.conv1(self.relu(x))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    out = self.conv2(self.relu(out))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    return torch.cat([x, out], 1)



class TransitionBlock(nn.Module):
  def __init__(self, in_planes, out_planes, dropRate=0.0):
    super(TransitionBlock, self).__init__()
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,padding=0, bias=False)
    self.droprate = dropRate
  def forward(self, x):
    out = self.conv1(self.relu(x))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    return F.upsample_nearest(out, scale_factor=2)



class TransitionBlock1(nn.Module):
  def __init__(self, in_planes, out_planes, dropRate=0.0):
    super(TransitionBlock1, self).__init__()
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,padding=0, bias=False)
    self.droprate = dropRate
  def forward(self, x):
    out = self.conv1(self.relu(x))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    return F.avg_pool2d(out, 2)



class TransitionBlock3(nn.Module):
  def __init__(self, in_planes, out_planes, dropRate=0.0):
    super(TransitionBlock3, self).__init__()
    self.relu = nn.ReLU(inplace=True)
    self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
    self.droprate = dropRate
  def forward(self, x):
    out = self.conv1(self.relu(x))
    if self.droprate > 0:
      out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
    return out