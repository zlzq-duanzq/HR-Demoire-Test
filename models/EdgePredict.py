import torch
import torch.nn as nn
from models.AFB import *

class EdgePredict(nn.Module):
    def __init__(self, in_c=6, out_c=3, args=None):
        super(EdgePredict, self).__init__()
        G, G0 = 64, 128
        kSize = 3

        self.encoding1 = nn.Conv2d(
            in_c, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.encoding2 = nn.Conv2d(
            G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        self.n_r = 1
        self.AFBs = nn.ModuleList()
        for i in range(self.n_r):
            self.AFBs.append(
                AFB_L1(growRate0=G0)
            )

        self.GFF = nn.Sequential(*[
            SELayer(self.n_r * G0),
            nn.Conv2d(self.n_r * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        self.decoding = nn.Sequential(*[
            nn.Conv2d(G0, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.Conv2d(G, out_c, kSize, padding=(kSize-1)//2, stride=1)
        ])

    def forward(self, x):
        ori = x
        x = ori

        f__1 = self.encoding1(x)
        x = self.encoding2(f__1)
        f__2 = x

        AFBs_out = []
        for i in range(self.n_r):
            x = self.AFBs[i](x)
            AFBs_out.append(x)

        x = self.GFF(torch.cat(AFBs_out, 1))
        x += f__2

        x = self.decoding(x)

        return x
