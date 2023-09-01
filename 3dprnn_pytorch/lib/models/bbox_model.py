import math
import copy
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import collections
# from torchvision.models.resnet import *
from lib.opts import *
# import maskrcnn_benchmark
from maskrcnn_benchmark.layers import ROIAlign
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.structures.bounding_box import BoxList
# from lib.model_detection.roi_layers.roi_align import ROIAlign


# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']


def download_resnets():
    model_urls = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }
    for key in model_urls:
        model_dict = model_zoo.load_url(model_urls[key])
        torch.save(model_dict, os.path.join(opt.resnet_dir, key+'.pth'))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetDiv(nn.Module):

    def __init__(self, block, layers, num_classes_reg=1000, num_classes_exist=1000):
        self.inplanes = 64
        super(ResNetDiv, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if opt.max_pool:
            self.maxpool2= nn.MaxPool2d(7, stride=1)
        else:
            self.avgpool = nn.AvgPool2d(7, stride=1)
        if opt.feat_bn:
            self.feature_bn = nn.BatchNorm1d(512 * block.expansion)
        if opt.stage == 'ssign':
            self.fc_reg = nn.Sequential(
                nn.Linear(512 * block.expansion, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1)
            )
        elif opt.add_fcs is not None:
            self.fc_reg = nn.Sequential(
                nn.Linear(512 * block.expansion, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_classes_reg)
            )
        else:
            self.fc_reg = nn.Linear(512 * block.expansion, num_classes_reg)
        if opt.exist:
            self.fc_exist = nn.Linear(512 * block.expansion, num_classes_exist)
        if 'idv' in opt.reg_init:
            self.fc_init_y = nn.Sequential(
                nn.Linear(512 * block.expansion, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, opt.xyz * 2)
            )
            self.fc_init_rot = nn.Sequential(
                nn.Linear(512 * block.expansion, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, opt.xyz * 1)
            )
            if opt.loss_c is not None:
                self.fc_init_cls = nn.Sequential(
                    nn.Linear(512 * block.expansion, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, opt.n_sem),
                    nn.Softmax(dim=1)
                )
            if opt.loss_box2d is not None:
                self.fc_init_box2d = nn.Sequential(
                    nn.Linear(512 * block.expansion, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 4)
                )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_out_size(self):
        if opt.out_r == '3dprnn':
            out_size = 4*opt.xyz + opt.dim_e
        elif opt.out_r == 'theta':
            out_size = 3*opt.xyz + opt.dim_e
        elif opt.out_r == 'class':
            out_size = 2*opt.xyz + 31 + opt.dim_e
        else:
            raise NotImplementedError
        if opt.loss_c is not None and opt.loss_box2d is not None:
            out_size = out_size + opt.n_sem + 4
        elif opt.loss_c is not None:
            out_size = out_size + opt.n_sem
        elif opt.loss_box2d is not None:
            out_size = out_size + 4
        else:
            out_size = out_size
        return out_size

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if opt.max_pool:
            x = self.maxpool2(x)
        else:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)   # [14, 512, 1, 1]
        if opt.feat_bn:
            x = self.feature_bn(x)
        x_reg = self.fc_reg(x)
        if opt.exist:
            x_exist = self.fc_exist(x)
            return x_reg, x_exist
        if 'idv' in opt.reg_init:
            y = self.fc_init_y(x)
            rot = self.fc_init_rot(x)
            cls, box2d = None, None
            if opt.loss_c is not None:
                cls = self.fc_init_cls(x)
            if opt.loss_box2d is not None:
                box2d = self.fc_init_box2d(x)
            x_init = (y, rot, cls, box2d)
            return x_reg, x_init
        if 'share' in opt.reg_init:
            return x_reg, x
        return x_reg


class FPN(nn.Module):

    def __init__(self, block, layers, num_classes_reg=1000, num_classes_exist=1000):
        self.inplanes = 64
        super(FPN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if opt.max_pool:
            self.maxpool2= nn.MaxPool2d(7, stride=1)
        else:
            self.avgpool = nn.AvgPool2d(7, stride=1)
        if opt.feat_bn:
            self.feature_bn = nn.BatchNorm1d(512 * block.expansion)
        self.fc_reg = nn.Linear(512 * block.expansion, num_classes_reg)
        if opt.exist:
            self.fc_exist = nn.Linear(512 * block.expansion, num_classes_exist)

        # FPN
        self.toplayer = nn.Conv2d(512 * block.expansion, 256,
                                  kernel_size=1, stride=1, padding=0)  # Reduce channels
        # self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(256 * block.expansion, 256,
                                   kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128 * block.expansion, 256,
                                   kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(64 * block.expansion, 256,
                                   kernel_size=1, stride=1, padding=0)
        # self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        # self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        # FPN
        # self.roi_align1 = ROIAlign((7, 7), 1.0 / 2.0, 2)
        # self.roi_align2 = ROIAlign((7, 7), 1.0 / 4.0, 2)
        # self.roi_align3 = ROIAlign((7, 7), 1.0 / 8.0, 2)
        # self.roi_align4 = ROIAlign((7, 7), 1.0 / 16.0, 2)
        # self.conv_final = nn.Conv2d(256, num_classes_reg, kernel_size=7, stride=1, padding=0)
        # ROI align

        if 'idv' in opt.reg_init:
            self.fc_init_y = nn.Sequential(
                nn.Linear(512 * block.expansion, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, opt.xyz * 2)
            )
            self.fc_init_rot = nn.Sequential(
                nn.Linear(512 * block.expansion, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, opt.xyz * 1)
            )
            if opt.loss_c is not None:
                self.fc_init_cls = nn.Sequential(
                    nn.Linear(512 * block.expansion, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, opt.n_sem),
                    nn.Softmax(dim=1)
                )
            if opt.loss_box2d is not None:
                self.fc_init_box2d = nn.Sequential(
                    nn.Linear(512 * block.expansion, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 4)
                )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, rois=None):
        # print('x', x.size())    # 224
        x = self.conv1(x)       # 112
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 56

        # FPN
        c1 = x                  # 56
        c2 = self.layer1(c1)    # 56
        c3 = self.layer2(c2)    # 28
        c4 = self.layer3(c3)    # 14
        c5 = self.layer4(c4)    # 7
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        # out = self.roi_align((p5, rois.view(-1, 5)))
        # out = self.conv_final(out)
        # local = [p2, p3, p4, p5]  # 56, 28, 14, 7
        local = [p2]

        x = c5
        if opt.max_pool:
            x = self.maxpool2(x)
        else:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)   # [14, 512, 1, 1]
        if opt.feat_bn:
            x = self.feature_bn(x)
        x_reg = self.fc_reg(x)
        if opt.exist:
            x_exist = self.fc_exist(x)
            global_encoding = (x_reg, x_exist)
        else:
            global_encoding = x_reg
        if 'global' in opt.feature_scale:
            global_encoding = global_encoding / global_encoding.norm(dim=1).unsqueeze(dim=1)
        if 'idv' in opt.reg_init:
            y = self.fc_init_y(x)
            rot = self.fc_init_rot(x)
            cls, box2d = None, None
            if opt.loss_c is not None:
                cls = self.fc_init_cls(x)
            if opt.loss_box2d is not None:
                box2d = self.fc_init_box2d(x)
            x_init = (y, rot, cls, box2d)
            return (global_encoding, local), x_init
        if 'share' in opt.reg_init:
            return (global_encoding, local), x

        return (global_encoding, local)


def fpn_division(model_name='resnet18', **kwargs):
    if model_name == 'resnet18':
        model = FPN(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif model_name == 'resnet34':
        model = FPN(BasicBlock, [3, 4, 6, 3], **kwargs)
    elif model_name == 'resnet50':
        model = FPN(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif model_name == 'resnet101':
        model = FPN(Bottleneck, [3, 4, 23, 3], **kwargs)
    elif model_name == 'resnet152':
        model = FPN(Bottleneck, [3, 8, 36, 3], **kwargs)
    else:
        raise NotImplementedError('Model is not defined!')

    return model


def resnet_division(model_name='resnet18', **kwargs):
    if model_name == 'resnet18':
        model = ResNetDiv(BasicBlock, [2, 2, 2, 2], **kwargs)
    elif model_name == 'resnet34':
        model = ResNetDiv(BasicBlock, [3, 4, 6, 3], **kwargs)
    elif model_name == 'resnet50':
        model = ResNetDiv(Bottleneck, [3, 4, 6, 3], **kwargs)
    elif model_name == 'resnet101':
        model = ResNetDiv(Bottleneck, [3, 4, 23, 3], **kwargs)
    elif model_name == 'resnet152':
        model = ResNetDiv(Bottleneck, [3, 8, 36, 3], **kwargs)
    else:
        raise NotImplementedError('Model is not defined!')

    return model


def load_parameters(model, model_dir):
    init_model_dict = model.state_dict()
    pre_model_dict = torch.load(model_dir)
    if opt.m_param_mode >= 0:
        print('=' * 20 + 'Load parameters: conv backbone.')
        # for key in init_model_dict.keys():
        #     if key not in ['fc_reg.weight', 'fc_reg.bias', 'fc_exist.weight', 'fc_exist.bias']:
        for key in pre_model_dict.keys():
            prefix = key.split('.')[0]
            if prefix not in ['fc', 'fc_reg', 'fc_exist']:
                init_model_dict[key] = copy.deepcopy(pre_model_dict[key])
    if opt.m_param_mode >= 1:
        print('=' * 20 + 'Load parameters: fc_reg.')
        # init_model_dict['fc_reg.weight'] = copy.deepcopy(pre_model_dict['fc.weight'])
        # init_model_dict['fc_reg.bias'] = copy.deepcopy(pre_model_dict['fc.bias'])
        for key in ['fc_reg.weight', 'fc_reg.bias']:
            init_model_dict[key] = copy.deepcopy(pre_model_dict[key])
    if opt.m_param_mode >= 2:
        print('=' * 20 + 'Load parameters: fc_exist.')
        for key in ['fc_exist.weight', 'fc_exist.bias']:
            init_model_dict[key] = copy.deepcopy(pre_model_dict[key])

    model.load_state_dict(init_model_dict)

    return model


def load_model_resnet(model):
    if opt.m_dir_mode == 0:
        print('=' * 20 + 'From scratch.')
        return model
    elif opt.m_dir_mode == 1:
        print('=' * 20 + 'Load from resnet dir.')
        model_dir = os.path.join(opt.resnet_dir, opt.model_name+'.pth')
    elif opt.m_dir_mode == 2:
        print('=' * 20 + 'Load from exp dir.')
        model_dir = os.path.join(opt.pre_dir, opt.pre_model)
    else:
        raise NotImplementedError
    model = load_parameters(model, model_dir)
    return model


def define_model_resnet():
    global opt
    opt = get_opt()
    part_dim = opt.n_sem*opt.n_para # 36  #**{'h':height, 'w':width})
    if opt.fpn:
        model = fpn_division(model_name=opt.model_name, num_classes_reg=part_dim, num_classes_exist=opt.n_sem)
    else:
        model = resnet_division(model_name=opt.model_name, num_classes_reg=part_dim, num_classes_exist=opt.n_sem)
    model = load_model_resnet(model)
    return model


def define_resnet_encoder(num_reg=None, num_exist=None):
    global opt
    opt = get_opt()
    if num_reg is None:
        num_reg = opt.n_sem * opt.n_para  # 36  #**{'h':height, 'w':width})
    if num_exist is None:
        num_exist = opt.n_sem
    if opt.fpn:
        model = fpn_division(model_name=opt.model_name, num_classes_reg=num_reg, num_classes_exist=num_exist)
    else:
        model = resnet_division(model_name=opt.model_name, num_classes_reg=num_reg, num_classes_exist=num_exist)
    return model
