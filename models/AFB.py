import torch
import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        #print(f"y: {y.shape}, b: {b}, c: {c}")

        # logger.info(str(y))
        # logger.info(f"{y.mean()}, {y.min()}, {y.max()}, {y.std()}")
        # embed()
        return x * y.expand_as(x)

class AFB_0(nn.Module):
    def __init__(self, Channels, kSize=3):
        super(AFB_0, self).__init__()
        Ch = Channels
        self.conv1 = nn.Conv2d(Ch, Ch, kSize, padding=(kSize-1)//2, stride=1)
        #TODO leaky
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        #self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(Ch, Ch, kSize, padding=(kSize-1)//2, stride=1)

    def forward(self, x):
        return x + self.relu(self.conv2(self.relu(self.conv1(x))))


class AFB_L1(nn.Module):
    def __init__(self, growRate0, kSize=3):
        super(AFB_L1, self).__init__()
        G0 = self.G0 = growRate0

        self.conv1 = AFB_0(G0)
        self.conv2 = AFB_0(G0)
        self.conv3 = AFB_0(G0)
        self.LFF = nn.Sequential(
            SELayer(G0+(self.G0)*2),
            nn.Conv2d(G0+(self.G0)*2, G0, 1, padding=0, stride=1),
        )

    def forward(self, x):
        res = []
        ox = x

        x = self.conv1(x)
        res.append(x)
        x = self.conv2(x)
        res.append(x)
        x = self.conv3(x)
        res.append(x)

        return self.LFF(torch.cat(res, 1)) + ox


class AFB_L2(nn.Module):
    def __init__(self, growRate0, kSize=3):
        super(AFB_L2, self).__init__()
        G0 = self.G0 = growRate0

        self.conv1 = AFB_L1(growRate0=G0)
        self.conv2 = AFB_L1(growRate0=G0)
        self.conv3 = AFB_L1(growRate0=G0)
        self.conv4 = AFB_L1(growRate0=G0)

        self.LFF = nn.Sequential(
            SELayer(G0+(self.G0)*3),
            nn.Conv2d(G0+(self.G0)*3, G0, 1, padding=0, stride=1),
        )

    def forward(self, x):
        res = []

        ox = x
        x = self.conv1(x)
        res.append(x)
        x = self.conv2(x)
        res.append(x)
        x = self.conv3(x)
        res.append(x)
        x = self.conv4(x)
        res.append(x)

        return self.LFF(torch.cat(res, 1)) + ox


class AFB_L3(nn.Module):
    def __init__(self, growRate0, kSize=3):
        super(AFB_L3, self).__init__()
        G0 = self.G0 = growRate0

        self.conv1 = AFB_L2(growRate0=G0)
        self.conv2 = AFB_L2(growRate0=G0)
        self.conv3 = AFB_L2(growRate0=G0)
        self.conv4 = AFB_L2(growRate0=G0)

        self.LFF = nn.Sequential(
            SELayer(G0+(self.G0)*3),
            nn.Conv2d(G0+(self.G0)*3, G0, 1, padding=0, stride=1),
        )

    def forward(self, x):
        res = []

        ox = x
        x = self.conv1(x)
        res.append(x)
        x = self.conv2(x)
        res.append(x)
        x = self.conv3(x)
        res.append(x)
        x = self.conv4(x)
        res.append(x)

        return self.LFF(torch.cat(res, 1)) + ox
