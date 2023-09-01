import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from lib.opts import *


class HGNet(nn.Module):
    def __init__(self, num_reg=None):
        super(HGNet, self).__init__()
        global opt
        opt = get_opt()
        #self.conv0 = nn.Conv2d(3, 64, 7, stride=2, padding=3).double()
        #self.bn0 = nn.BatchNorm2d(64)
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            #nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128, 128),
            ResidualBlock(128, opt.nFeats)
            )
        self.hg = nn.ModuleList([Hourglass(4, opt.nFeats) for i in range(opt.nStack)])
        self.res = nn.ModuleList([ResidualBlock(opt.nFeats, opt.nFeats) for i in range(opt.nStack*opt.nModules)])
        self.convblock = nn.ModuleList([ConvLayer(opt.nFeats, opt.nFeats) for i in range(opt.nStack)])
        self.conv1 = nn.ModuleList([nn.Conv2d(opt.nFeats, opt.nClasses, 1) for i in range(opt.nStack)])
        self.conv2 = nn.ModuleList([nn.Conv2d(opt.nFeats, opt.nFeats, 1) for i in range(opt.nStack-1)])
        self.conv3 = nn.ModuleList([nn.Conv2d(opt.nClasses, opt.nFeats, 1) for i in range(opt.nStack-1)])

        if opt.stage == 'prnn':
            self.conv2_0 = nn.Conv2d(opt.nFeats, opt.nFeats, 1)
            if opt.sem_map is not None:
                if opt.sem_map == 'all':
                    self.conv3_0 = nn.Conv2d(opt.nClasses, opt.nFeats, 1)
                    self.heatmap_start = 0
                    self.heatmap_end = opt.nClasses
                else:
                    self.conv3_0 = nn.Conv2d(1, opt.nFeats, 1)
                    self.heatmap_start = int(opt.sem_map) - 1
                    self.heatmap_end = int(opt.sem_map)
            self.conv_seq = nn.Sequential(
                nn.Conv2d(opt.nFeats, opt.nFeats, 7, stride=4, padding=3),
                nn.Conv2d(opt.nFeats, opt.nFeats, 7, stride=4, padding=3),
                nn.Conv2d(opt.nFeats, opt.nFeats, 7, stride=4, padding=3)
            )
            # self.fc = nn.Linear(512, num_reg)

    def forward(self, x):
        global opt
        opt = get_opt()
        out = collections.OrderedDict()
        #x = self.conv0(x)
        #x = self.bn0(x)
        inter = self.initial(x)
        for i in range(opt.nStack):
            # import pdb
            # pdb.set_trace()
            hgout = self.hg[i](inter)
            # residual layers at output resolution
            for j in range(opt.nModules):
                hgout = self.res[i*opt.nModules+j](hgout)
            # linear layer to produce first set of predictions
            hgout = self.convblock[i](hgout)
            # predicted heatmaps
            tmpout = self.conv1[i](hgout)
            out[i] = tmpout
            # add predictions back
            if i < opt.nStack - 1:
                hgout_ = self.conv2[i](hgout)
                tmpout_ = self.conv3[i](tmpout)
                inter = inter + hgout_ + tmpout_
            if opt.stage == 'prnn' and i == opt.nStack - 1:
                hgout_ = self.conv2_0(hgout)
                if opt.sem_map is not None:
                    tmpout_ = self.conv3_0(tmpout[:,self.heatmap_start:self.heatmap_end,:,:])
                    inter = inter + hgout_ + tmpout_    # [14, 256, 64, 64]
                else:
                    inter = inter + hgout_
                inter = self.conv_seq(inter)
                inter = inter.view(inter.size(0), -1)
                # import pdb;pdb.set_trace()
                # inter = self.fc(inter)
                return inter

        return out


# Apply 1x1 convolution, stride 1, no padding
class ConvLayer(nn.Module):
    def __init__(self, InChannels, OutChannels):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(InChannels, OutChannels, 1)
        self.bn = nn.BatchNorm2d(OutChannels)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.bn(self.conv(x)))
        #output = self.relu(self.conv(x))
        return output


class ResidualBlock(nn.Module):
    def __init__(self, InChannels, OutChannels):
        super(ResidualBlock, self).__init__()
        self.ConvBlock = nn.Sequential(
            nn.BatchNorm2d(InChannels),
            nn.ReLU(),
            nn.Conv2d(InChannels, OutChannels//2, 1),
            nn.BatchNorm2d(OutChannels//2),
            nn.ReLU(),
            nn.Conv2d(OutChannels//2, OutChannels//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(OutChannels//2),
            nn.ReLU(),
            nn.Conv2d(OutChannels//2, OutChannels, 1)
            )
        self.SkipLayer = nn.Conv2d(InChannels, OutChannels, 1)
        self.InChannels = InChannels
        self.OutChannels = OutChannels

    def forward(self, x):
        conv = self.ConvBlock(x)
        if self.InChannels == self.OutChannels:
            skip = x
        else:
            skip = self.SkipLayer(x)
        output = conv + skip
        return output


class Hourglass(nn.Module):
    def __init__(self, n, nfeats):
        super(Hourglass, self).__init__()
        self.residual1 = nn.ModuleList([ResidualBlock(nfeats, nfeats) for i in range(opt.nModules)])
        self.residual2 = nn.ModuleList([ResidualBlock(nfeats, nfeats) for i in range(opt.nModules)])
        self.residual3 = nn.ModuleList([ResidualBlock(nfeats, nfeats) for i in range(opt.nModules)])
        self.residual4 = nn.ModuleList([ResidualBlock(nfeats, nfeats) for i in range(opt.nModules)])
        self.maxpool = nn.MaxPool2d(2)
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)
        # self.upsampling = nn.functional.interpolate(scale_factor=2)
        self.n = n
        if n > 1:
            self.hg = Hourglass(n-1, nfeats)

    def forward(self, x):
        global opt
        opt = get_opt()
        # uppper branch
        up1 = x
        for i in range(opt.nModules):
            up1 = self.residual1[i](up1)

        # lower branch
        low1 = self.maxpool(x)
        for i in range(opt.nModules):
            low1 = self.residual2[i](low1)
        if self.n > 1:
            low2 = self.hg(low1)
        else:
            low2 = low1
            for i in range(opt.nModules):
                low2 = self.residual3[i](low2)
        low3 = low2
        for i in range(opt.nModules):
            low3 = self.residual4[i](low3)
        up2 = self.upsampling(low3)

        # bring two branches together
        output = up1 + up2
        return output
