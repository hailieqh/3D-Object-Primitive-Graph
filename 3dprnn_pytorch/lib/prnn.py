import os
import time
import copy
import math
import random
import pprint
import scipy.io
import numpy as np
import torchfile
import pdb

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from colorama import Fore, Back, Style
from scipy.spatial.distance import directed_hausdorff

# from lib.datasets.dataset import SYNSet, SYNTestSet
# from lib.utils.statistics import StatisticsPrnn
from lib.models import PRNNModel, MixtureCriterion, Faster3DProposal, BBoxCriterion, MaskCriterion, PrimGraphModel
from lib.models import VOXAEnet, define_resnet_encoder, HGNet, define_model_resnet
from lib.datasets import SYNSet, SYNTestSet, SYNSaveNNSet
from lib.datasets import Pix3DChair, RandomRColor, PadSquare, RandomRotate, RandomHorizontalFlip, Rescale, ToTensor
from lib.utils import StatisticsPrnn
from lib.utils.prnn_utils import *
from lib.utils.test_utils import *
from lib.opts import *


class PRNN(object):
    def __init__(self, cfg, logger):
        super(PRNN, self).__init__()
        global opt
        opt = get_opt()
        # self.cfg = cfg
        # self.logger = logger

        # logger.info(pprint.pformat(cfg, indent=4))
        # if not (opt.cal_iou or opt.init_torch):
        self.phases = None
        self.root_dir = opt.data_dir
        self.model = self.build_model()
        self.optimizer, self.scheduler = self.build_optimizer()
        self.mix_criterion, self.rot_criterion, self.cls_criterion, \
        self.bbox_criterion, self.st_criterion, self.e_criterion, self.box2d_criterion = self.build_criterion()
        self.data_loaders, self.data_sizes, self.datasets = self.build_dataloader()
        if opt.use_gpu:
            self.set_gpu()

    def set_gpu(self):
        # print(opt.GPU)
        # torch.cuda.set_device(opt.GPU)
        self.model = self.model.cuda()
        self.mix_criterion = self.mix_criterion.cuda()
        self.rot_criterion = self.rot_criterion.cuda()
        self.cls_criterion = self.cls_criterion.cuda()
        self.bbox_criterion = self.bbox_criterion.cuda()
        self.st_criterion = self.st_criterion.cuda()
        self.e_criterion = self.e_criterion.cuda()
        self.box2d_criterion = self.box2d_criterion.cuda()

    def build_model(self):
        if opt.stage == 'ssign':
            if opt.encoder == 'depth_new':
                model = VOXAEnet(out_size=1)
            else:
                model = define_model_resnet()
            if opt.demo is not None:# or opt.save_nn:
                print('=' * 40, 'Loading pretrained model to test...')
                if int(opt.GPU) == -1:
                    model = nn.DataParallel(model)
                if int(opt.model_epoch) != -1:  # best_model.pth here
                    opt.pre_model = 'model_epoch_{}.pth'.format(opt.model_epoch)
                model_dict = torch.load(os.path.join(opt.exp_dir, opt.pre_model))
                model.load_state_dict(model_dict)
            else:
                if opt.encoder != 'depth_new':
                    print('=' * 40, 'Loading pretrained resnet model to train...')
                    if opt.fix_encoder:
                        # for name, param in model.prnn_core.rgb_encoder.named_parameters():
                        for name, param in model.named_parameters():
                            # print(name)
                            prefix = name.split('.')[0]
                            if prefix not in ['fc_reg', 'feature_bn']:
                                # print(name)
                                param.requires_grad = False
                if int(opt.GPU) == -1:
                    model = nn.DataParallel(model)
        elif opt.save_w_vector:
            print('=' * 40, 'Saving w_vectors from pretrained model...')
            if opt.encoder in ['depth', 'depth_new']:
                model = VOXAEnet()
            elif opt.encoder == 'resnet':
                model = define_resnet_encoder(num_reg=opt.con_size)
            elif opt.encoder == 'hg':
                model = HGNet(num_reg=opt.con_size)
            else:
                raise NotImplementedError
            init_model_dict = model.state_dict()
            pre_model_dict = torch.load(os.path.join(opt.exp_dir, opt.pre_model))
            for key in pre_model_dict.keys():
                if opt.encoder in ['depth', 'depth_new']:
                    # if key[:13] == 'depth_encoder':
                    if 'depth_encoder' in key:
                        idx = key.index('depth_encoder') + 14
                        init_model_dict[key[idx:]] = copy.deepcopy(pre_model_dict[key])
                elif opt.encoder == 'resnet' or opt.encoder == 'hg':
                    # if key[:11] == 'rgb_encoder':
                    if 'rgb_encoder' in key:
                        idx = key.index('rgb_encoder') + 12
                        init_model_dict[key[idx:]] = copy.deepcopy(pre_model_dict[key])
            model.load_state_dict(init_model_dict)
            if int(opt.GPU) == -1:
                model = nn.DataParallel(model)
        elif opt.bi_lstm == 'online':
            model = PrimGraphModel()
            if opt.demo is not None:
                print('=' * 40, 'Loading pretrained model to test...')
                if int(opt.GPU) == -1:
                    model = nn.DataParallel(model)
                model_dict = torch.load(os.path.join(opt.exp_dir, opt.pre_model))
                model.load_state_dict(model_dict)
                model_params = model.named_parameters()
                for name, param in model_params:
                    param.requires_grad = False
            else:
                print('=' * 40, 'Loading pretrained lstms to train...')
                init_model_dict = model.state_dict()
                pre_model_dict = {x: torch.load(opt.pre_dir[x]) for x in ['forward', 'backward']}
                print('*' * 30 + 'init model keys')
                print(init_model_dict.keys())
                print('*' * 30 + 'pre model keys')
                print(pre_model_dict.keys())
                print('*' * 30 + 'load params')
                for x in ['forward', 'backward']:
                    for key in pre_model_dict[x].keys():
                        # prefix = key.split('.')[0]
                        print(x, key)
                        init_model_dict['prnn_model_{}.'.format(x) + key] = copy.deepcopy(pre_model_dict[x][key])
                model.load_state_dict(init_model_dict)
                if opt.fix_encoder:
                    print('*' * 30 + 'fix params')
                    model_params = model.named_parameters()
                    for name, param in model_params:
                        prefix = name.split('.')[0]
                        print(param.sum(), param.size())
                        print(prefix, name)
                        if prefix in ['prnn_model_forward', 'prnn_model_backward']:
                            param.requires_grad = False
                            print('     ', prefix, name)
                if int(opt.GPU) == -1:
                    model = nn.DataParallel(model)
        else:
            if 'proposal' in opt.faster:
                model = Faster3DProposal()
            else:
                model = PRNNModel(inverse=opt.inverse)
            if opt.demo is not None:
                print('=' * 40, 'Loading pretrained model to test...')
                if int(opt.GPU) == -1:
                    model = nn.DataParallel(model)
                model_dict = torch.load(os.path.join(opt.exp_dir, opt.pre_model))
                model.load_state_dict(model_dict)
            elif opt.encoder == 'resnet' or opt.encoder == 'hg':
                print('=' * 40, 'Loading pretrained resnet model to train...')
                init_model_dict = model.state_dict()
                pre_model_dict = torch.load(opt.pre_dir)
                print('opt.pre_dir', opt.pre_dir)
                # print('*' * 30 + 'init model keys')
                # print(init_model_dict.keys())
                # print('*' * 30 + 'pre model keys')
                # print(pre_model_dict.keys())
                # print('*' * 30 + 'load params')
                for key in pre_model_dict.keys():
                    prefix = key.split('.')[0]
                    print(key, prefix)
                    if prefix not in ['fc_reg', 'fc_exist', 'fc', 'feature_bn']:
                        print('     ', key)
                        if opt.refine is not None and opt.pre_dir.split('/')[-2] != 'resnet':
                            init_model_dict[key] = copy.deepcopy(pre_model_dict[key])
                        elif opt.re_lstm != 0:
                            if opt.bg_lstm:
                                if 'cls_linear' in key.split('.'):
                                    print('     ', key)
                                    continue
                            init_model_dict[key] = copy.deepcopy(pre_model_dict[key])
                        else:
                            init_model_dict['rgb_encoder.' + key] = copy.deepcopy(pre_model_dict[key])
                model.load_state_dict(init_model_dict)
                if opt.fix_encoder:
                    print('*' * 30 + 'fix params')
                    # for name, param in model.prnn_core.rgb_encoder.named_parameters():
                    if opt.refine is not None and opt.fix_prnn == 'yes':
                        model_params = model.named_parameters()
                    elif opt.re_lstm != 0:
                        model_params = model.named_parameters()
                    else:
                        model_params = model.rgb_encoder.named_parameters()
                    for name, param in model_params:
                        # prefix = name.split('.')[0]
                        # if prefix not in ['fc_reg', 'fc_exist', 'feature_bn',
                        #                 'conv2_0', 'conv3_0', 'conv_seq',
                        #                 'fc']:
                        if opt.refine is not None and opt.fix_prnn == 'no':
                            if opt.pre_dir.split('/')[-2] != 'resnet':
                                name = 'rgb_encoder.' + name    ## freeze the whole rgb_encoder
                        print(param.sum(), param.size())
                        print(name)
                        if opt.re_lstm > 0:
                            inter_name = name.split('.')[1]
                            if 'linear' in inter_name:
                                print('*'*10, name)
                                continue
                            if 'linear' in name:
                                print('*'*100)
                        if opt.re_lstm < 0:
                            if 'prnn_core' in name:
                                print('*' * 10, name)
                                continue
                        if name in pre_model_dict.keys():
                            param.requires_grad = False
                            print('     ', name)
                if int(opt.GPU) == -1:
                    model = nn.DataParallel(model)
        return model

    def build_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, betas=(0.95, 0.999), # alpha=0.95
                                     eps=opt.eps, weight_decay=opt.weight_decay, amsgrad=False)
        scheduler = None
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)
        return optimizer, scheduler

    def update_optimizer(self):
        rot_params = [p[1] for p in self.model.named_parameters() if p[0].split('.')[1] == 'rot_linear']
        other_params = [p[1] for p in self.model.named_parameters() if p[0].split('.')[1] != 'rot_linear']
        self.optimizer = torch.optim.Adam([
            {'params': rot_params, 'lr': opt.lr_r},
            {'params': other_params},
        ], lr=opt.lr, betas=(0.95, 0.999), eps=opt.eps, weight_decay=opt.weight_decay, amsgrad=False) # alpha=0.95

    def build_criterion(self):
        mix_criterion = MixtureCriterion()
        mix_criterion.set_size_average()
        bbox_criterion = BBoxCriterion()

        if opt.loss_c == 'nllc' or opt.len_adjust:
            cls_criterion = MaskCriterion(metric='nllc')
            cls_criterion.set_size_average()
        else:
            cls_criterion = nn.NLLLoss()

        if opt.loss_y == 'l2':
            st_criterion = nn.MSELoss()
        elif opt.loss_y == 'l1':
            st_criterion = nn.L1Loss()
        elif opt.loss_y in ['l2c', 'l1c']:
            st_criterion = MaskCriterion(metric=opt.loss_y)
            st_criterion.set_size_average()
        else:
            st_criterion = nn.MSELoss()

        if opt.loss_e == 'bce':
            e_criterion = nn.BCELoss()
        elif opt.loss_e == 'bcec':
            e_criterion = MaskCriterion(metric=opt.loss_e)
            e_criterion.set_size_average()
        elif opt.loss_e == '3dprnn':
            e_criterion = MixtureCriterion()
            e_criterion.set_size_average()
        elif opt.loss_e == 'l2':
            e_criterion = nn.MSELoss()
        elif opt.loss_e == 'l1':
            e_criterion = nn.L1Loss()
        else:
            e_criterion = nn.BCELoss()

        if opt.loss_r == 'l2':
            rot_criterion = nn.MSELoss()
        elif opt.loss_r == 'l1':
            rot_criterion = nn.L1Loss()
        elif opt.loss_r in ['l2c', 'l1c', 'nllc']:
            rot_criterion = MaskCriterion(metric=opt.loss_r)
            rot_criterion.set_size_average()
        else:
            rot_criterion = nn.MSELoss()

        if opt.loss_box2d == 'sl1c':
            box2d_criterion = MaskCriterion(metric=opt.loss_box2d)
            box2d_criterion.set_size_average()
        else:
            box2d_criterion = nn.SmoothL1Loss()

        return mix_criterion, rot_criterion, cls_criterion, bbox_criterion, st_criterion, e_criterion, box2d_criterion

    def build_dataloader(self):
        return self.build_dataloader_prnn()

    def build_dataloader_bbox(self):
        if opt.save_w_vector:
            self.phases = ['train', 'val'] + ['test_'+x for x in opt.file_names['obj_classes']]
            batches = {x: 1 for x in self.phases}
        else:
            if opt.demo is None:
                if opt.train_val:
                    self.phases = ['train_val']
                else:
                    self.phases = ['train', 'val']
                batches = {'train': opt.train_batch, 'val': opt.valid_batch, 'train_val': opt.train_batch}
            else:
                self.phases = ['test_'+x for x in opt.file_names['obj_classes']]
                batches = {x: opt.test_batch for x in self.phases}
        # self.phases = ['train', 'val']
        data_transform = {x: transforms.Compose([PadSquare(), Rescale((opt.input_res, opt.input_res)), ToTensor()]) for x in self.phases}
        is_shuffle = {x: False for x in self.phases}
        if not opt.save_w_vector:
            for phase in ['train', 'train_val']:
                if phase in self.phases:
                    data_transform[phase] = transforms.Compose([
                        RandomRColor(),
                        PadSquare(),
                        RandomRotate(),
                        Rescale((opt.input_res, opt.input_res)),
                        ToTensor()])
                    is_shuffle[phase] = bool((opt.demo is None)*opt.shuffle)
        datasets = {x: Pix3DChair(transform=data_transform[x], phase=x) for x in self.phases}
        # batches = {'train': opt.train_batch, 'val': opt.valid_batch, 'test': opt.test_batch}
        data_loaders = {x: DataLoader(datasets[x], batch_size=batches[x], shuffle=is_shuffle[x],
                                      drop_last=True, num_workers=opt.num_workers) for x in self.phases}
        data_sizes = {x: (len(datasets[x]) - len(datasets[x]) % batches[x]) for x in self.phases}

        return data_loaders, data_sizes, datasets

    def build_dataloader_prnn(self):
        # train_set, val_set, test_set = SpeechDataset.splits(self.cfg)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if opt.encoder == 'resnet':
            transformer = []
            # transformer = [normalize]
        else:
            transformer = []
        if opt.save_w_vector:
            self.phases = ['train', 'val'] + ['test_'+x for x in opt.file_names['obj_classes']]
            # self.phases = ['train', 'val', 'test_chair', 'test_table', 'test_night_stand']
            transformer = [PadSquare(), Rescale((opt.input_res, opt.input_res)), ToTensor()] + transformer
            data_transform = {x: transforms.Compose(transformer) for x in self.phases}
            datasets = {x: SYNSaveNNSet(inverse=opt.inverse, transform=data_transform[x], phase=x) for x in self.phases}
            batches = {x: 1 for x in self.phases}
            is_shuffle = {x: False for x in self.phases}
            drop_flag = False
        else:
            if opt.demo is None:
                if (opt.encoder == 'resnet' or opt.encoder == 'hg') and opt.obj_class[:6] != '3dprnn':
                    train_transformer = [RandomRColor(), PadSquare(), Rescale((opt.input_res, opt.input_res)),
                                   # RandomRColor(), PadSquare(), RandomRotate(), Rescale((opt.input_res, opt.input_res)),
                                   # RandomHorizontalFlip(), Rescale((opt.input_res, opt.input_res)),
                                   # PadSquare(), Rescale((opt.input_res, opt.input_res)), RandomRColor(),
                                   # RandomRotate(), RandomHorizontalFlip(),
                                   ToTensor()] + transformer
                else:
                    train_transformer = [PadSquare(), Rescale((opt.input_res, opt.input_res)),
                                         ToTensor()] + transformer
                val_transformer = [PadSquare(), Rescale((opt.input_res, opt.input_res)), ToTensor()] + transformer
                train_transform = transforms.Compose(train_transformer)
                data_transform = {}
                if opt.train_val:
                    self.phases = ['train_val']
                    data_transform['train_val'] = train_transform
                else:
                    self.phases = ['train', 'val']
                    data_transform['train'] = train_transform
                    data_transform['val'] = transforms.Compose(val_transformer)
                datasets = {x: SYNSet(inverse=opt.inverse, transform=data_transform[x], phase=x) for x in self.phases}
                batches = {'train': opt.train_batch, 'val': opt.valid_batch, 'train_val': opt.train_batch}
                is_shuffle = {'train': opt.shuffle, 'val': False, 'train_val': opt.shuffle}
                drop_flag = False
            else:
                self.phases = ['test_'+x for x in opt.file_names['obj_classes']]
                if opt.stage == 'ssign':
                    if opt.stack_learn in ['a', 'b']:
                        self.phases = ['val']
                        opt.test_version = '2'
                    else:
                        if 'nyu' not in opt.obj_class:
                            self.phases += ['train', 'val']
                        opt.test_version = 'ts'
                # self.phases = ['test_chair', 'test_table', 'test_night_stand']
                transformer = [PadSquare(), Rescale((opt.input_res, opt.input_res)), ToTensor()] + transformer
                data_transform = {x: transforms.Compose(transformer) for x in self.phases}
                if not opt.gt_in and opt.stage != 'ssign' and 'proposal' not in opt.faster:
                    datasets ={x: SYNTestSet(inverse=opt.inverse, transform=data_transform[x], phase=x) for x in self.phases}
                else:
                    datasets = {x: SYNSet(inverse=opt.inverse, transform=data_transform[x], phase=x) for x in self.phases}
                batches = {x: opt.test_batch for x in self.phases}
                is_shuffle = {x: False for x in self.phases}
                drop_flag = False
        data_loaders = {x: DataLoader(datasets[x], batch_size=batches[x], shuffle=is_shuffle[x],
                                      drop_last=drop_flag, num_workers=opt.num_workers) for x in self.phases}
        data_sizes = {x: (len(datasets[x]) - len(datasets[x]) % batches[x] * drop_flag) for x in self.phases}
        # data_sizes is related to drop_last

        return data_loaders, data_sizes, datasets

    def build_dataloader_prnn_test(self):
        transformer = []
        phases = ['test_' + x for x in opt.file_names['obj_classes']]
        if opt.stage == 'ssign':
            if opt.stack_learn in ['a', 'b']:
                phases = ['val']
                opt.test_version = '2'
            else:
                phases += ['train', 'val']
                opt.test_version = 'ts'
        # self.phases = ['test_chair', 'test_table', 'test_night_stand']
        transformer = [PadSquare(), Rescale((opt.input_res, opt.input_res)), ToTensor()] + transformer
        data_transform = {x: transforms.Compose(transformer) for x in phases}
        if not opt.gt_in and opt.stage != 'ssign':
            datasets = {x: SYNTestSet(inverse=opt.inverse, transform=data_transform[x], phase=x) for x in phases}
        else:
            datasets = {x: SYNSet(inverse=opt.inverse, transform=data_transform[x], phase=x) for x in phases}
        batches = {x: opt.test_batch for x in phases}
        is_shuffle = {x: False for x in phases}
        drop_flag = False

        data_loaders = {x: DataLoader(datasets[x], batch_size=batches[x], shuffle=is_shuffle[x],
                                      drop_last=drop_flag, num_workers=opt.num_workers) for x in phases}
        data_sizes = {x: (len(datasets[x]) - len(datasets[x]) % batches[x] * drop_flag) for x in phases}
        # data_sizes is related to drop_last

        return data_loaders, data_sizes, datasets

    def train_ssign(self):
        statis = StatisticsPrnn()
        for epoch in range(opt.n_epochs + 1):
            statis.print_epoch_begin(epoch)
            save_length_pred = {}
            save_length_gt = {}
            for phase in self.phases:
                if opt.fix_bn:
                    if phase == 'train' or phase == 'train_val':
                        self.model.train(False)
                        if opt.feat_bn:
                            self.model.feature_bn.train()
                        # self.model.rgb_encoder.eval()
                        # self.model.rgb_encoder.fc_reg.train()
                        # self.model.rgb_encoder.fc_exist.train()
                    else:
                        self.model.train(False)
                else:
                    if phase == 'train' or phase == 'train_val':
                        self.model.train(True)
                    else:
                        self.model.train(False)
                statis.reset_epoch(phase)
                i_batch = -1
                save_length_pred[phase] = np.ones((self.data_sizes[phase], 1))
                save_length_gt[phase] = np.ones((self.data_sizes[phase], 1))
                print(self.data_sizes)
                for i_batch, data in enumerate(self.data_loaders[phase]):
                    labels, _, _, _, depth_mat, _, _, _, _, _ = data
                    labels, depth_mat = labels.float(), depth_mat.float()
                    if opt.use_gpu:
                        labels, depth_mat = labels.cuda(), depth_mat.cuda()
                    if opt.zero_in:
                        depth_mat = torch.ones(depth_mat.size()).cuda() * 0.
                    outputs = self.model(depth_mat)
                    loss = self.e_criterion(outputs, labels)
                    preds = torch.round(outputs)
                    more = torch.sum(preds > labels).type(torch.FloatTensor) / preds.size()[0]
                    less = torch.sum(preds < labels).type(torch.FloatTensor) / preds.size()[0]
                    equal = torch.sum(preds == labels).type(torch.FloatTensor) / preds.size()[0]
                    if epoch == 6:
                        if phase == 'train':
                            start_id = i_batch * opt.train_batch
                        else:
                            start_id = i_batch * opt.valid_batch
                        print(start_id)
                        print(preds.size())
                        save_length_pred[phase][start_id : start_id + preds.size()[0], :] = preds.cpu().detach().numpy()
                        save_length_gt[phase][start_id: start_id + labels.size()[0], :] = labels.cpu().detach().numpy()
                    error = torch.mean(torch.abs(preds - labels) / labels)
                    if (phase == 'train' or phase == 'train_val') and epoch > 0:# and not opt.save_nn:
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    statis.accumulate(loss, more, less, equal, loss*0, error, loss*0, loss*0)
                    # if i_batch % 10 == 0:
                    statis.print_batch(epoch, i_batch)
                if None: #epoch == 6:
                    scipy.io.savemat('{}/length_pred_{}.mat'.format(opt.exp_dir, phase),
                                     {'length_pred': save_length_pred[phase]})
                    scipy.io.savemat('{}/length_gt_{}.mat'.format(opt.exp_dir, phase),
                                     {'length_gt': save_length_gt[phase]})
                is_best_tag = statis.is_best(i_batch, phase)
                if is_best_tag:# and not opt.save_nn:
                    best_model = copy.deepcopy(self.model)
                    torch.save(best_model.cpu().state_dict(), '{}/best_model.pth'.format(opt.exp_dir))
                    print('=' * 40, 'update best model')
                statis.print_epoch_end(epoch, phase)
                if opt.visdom:
                    statis.plot(epoch, phase, is_best_tag)
            if epoch % opt.snapshot == 0:# and not opt.save_nn:
                snapshot = copy.deepcopy(self.model)
                torch.save(snapshot.cpu().state_dict(), '{}/model_epoch_{}.pth'.format(opt.exp_dir, epoch))
        statis.print_final()
        return

    def test_ssign(self):
        save_length_pred = {}
        save_length_gt = {}
        for phase in self.phases:
            self.model.train(False)
            i_batch = -1
            save_length_pred[phase] = np.ones((self.data_sizes[phase], 1))
            save_length_gt[phase] = np.ones((self.data_sizes[phase], 1))
            loss_sum, more_sum, less_sum, equal_sum, error_sum = 0, 0, 0, 0, 0
            print(self.data_sizes)
            for i_batch, data in enumerate(self.data_loaders[phase]):
                labels, _, _, _, depth_mat, _, _, _, _, _ = data
                labels, depth_mat = labels.float(), depth_mat.float()
                if opt.use_gpu:
                    labels, depth_mat = labels.cuda(), depth_mat.cuda()
                if opt.zero_in:
                    depth_mat = torch.ones(depth_mat.size()).cuda() * 0.
                outputs = self.model(depth_mat)
                loss = self.e_criterion(outputs, labels)
                preds = torch.round(outputs)
                more = torch.sum(preds > labels).type(torch.FloatTensor) / preds.size()[0]
                less = torch.sum(preds < labels).type(torch.FloatTensor) / preds.size()[0]
                equal = torch.sum(preds == labels).type(torch.FloatTensor) / preds.size()[0]
                # if phase == 'train':
                #     start_id = i_batch * opt.train_batch
                # else:
                #     start_id = i_batch * opt.valid_batch
                # print(start_id, preds.size())
                # save_length_pred[phase][start_id : start_id + preds.size()[0], :] = preds.cpu().detach().numpy()
                # save_length_gt[phase][start_id: start_id + labels.size()[0], :] = labels.cpu().detach().numpy()
                save_length_pred[phase][i_batch, :] = preds.cpu().detach().numpy()
                save_length_gt[phase][i_batch, :] = labels.cpu().detach().numpy()
                error = torch.mean(torch.abs(preds - labels) / labels)
                loss_sum += loss.item()
                more_sum += more.item()
                less_sum += less.item()
                equal_sum += equal.item()
                error_sum += error.item()
            print(phase, loss_sum/(i_batch+1), more_sum/(i_batch+1), less_sum/(i_batch+1),
                  equal_sum/(i_batch+1), error_sum/(i_batch+1))
            if opt.stack_learn in ['a', 'b']:
                scipy.io.savemat('{}/length_pred_{}_{}.mat'.format(opt.exp_dir, phase, opt.stack_learn),
                                 {'length_pred': save_length_pred[phase]})
                scipy.io.savemat('{}/length_gt_{}_{}.mat'.format(opt.exp_dir, phase, opt.stack_learn),
                                 {'length_gt': save_length_gt[phase]})
            else:
                scipy.io.savemat('{}/length_pred_{}.mat'.format(opt.exp_dir, phase),
                                 {'length_pred': save_length_pred[phase]})
                scipy.io.savemat('{}/length_gt_{}.mat'.format(opt.exp_dir, phase),
                                 {'length_gt': save_length_gt[phase]})
        return

    def compute_loss_sequence(self, outputs, targets, c_mask_mat, losses):
        loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d, loss_r_theta, loss_r_axis = losses
        seq_len = len(outputs) + 1  # the last one is the complete sequence of x
        assert seq_len == len(targets) + 1  # and seq_len == len(outputs[seq_len - 1])
        for i in range(seq_len - 1):    # 0, 1, 2, ..., 3n - 2
            if 'compute_loss' in opt.check:
                import pdb;pdb.set_trace()
            c_mask_i = i + 1
            if opt.reg_init != 'None':
                c_mask_i = i
            if opt.stop_sign_w and i == seq_len - 2:
                weight_e = seq_len - 2.
            else:
                weight_e = 1.
            if opt.loss_y == 'gmm':
                self.mix_criterion.set_mask(c_mask_mat[:, :, c_mask_i])
                loss_xe = self.mix_criterion(outputs[i][0], targets[i][0])
                loss_x += loss_xe[0]
                loss_e += loss_xe[1] * weight_e
                loss_s += loss_xe[2]
            elif opt.loss_y in ['l2', 'l2c', 'l1', 'l1c']:
                if opt.loss_y in ['l2c', 'l1c']:
                    self.st_criterion.set_mask(c_mask_mat[:, :, c_mask_i])
                if opt.loss_e is None:
                    loss_x += self.st_criterion(outputs[i][0], targets[i][0])
                else:
                    loss_x += self.st_criterion(outputs[i][0][:, opt.dim_e:], targets[i][0][:, :2*opt.xyz])

                if opt.loss_e in ['bcec', '3dprnn']:
                    self.e_criterion.set_mask(c_mask_mat[:, :, c_mask_i])
                if opt.loss_e is None:
                    loss_e += 0
                elif opt.loss_e == '3dprnn' and opt.loss_y != 'gmm':
                    loss_e += self.e_criterion(outputs[i][0], targets[i][0]) * weight_e
                else:
                    loss_e += self.e_criterion(outputs[i][0][:, :opt.dim_e], targets[i][0][:, 2*opt.xyz:]) * weight_e
            if opt.loss_r in ['l2c', 'l1c', 'nllc']:
                self.rot_criterion.set_mask(c_mask_mat[:, :, c_mask_i])
            if opt.out_r == 'class':
                loss_r_axis += self.rot_criterion(torch.log(outputs[i][1][:, 27:]), targets[i][1][:, 1])
                theta_mask = targets[i][1][:, 1].unsqueeze(dim=0).transpose(0, 1).float() * c_mask_mat[:, :, c_mask_i]
                theta_mask = (theta_mask > 0).float()
                if torch.sum(theta_mask).item() > 0:
                    self.rot_criterion.set_mask(theta_mask)
                    loss_r_theta += self.rot_criterion(torch.log(outputs[i][1][:, :27]), targets[i][1][:, 0])
            else:
                loss_r += self.rot_criterion(outputs[i][1], targets[i][1])

            if opt.loss_c in ['nllc'] or opt.len_adjust:
                self.cls_criterion.set_mask(c_mask_mat[:, :, c_mask_i])
            if opt.loss_c is not None:
                if (i + opt.step_start) % (3 // opt.xyz) == 0:
                    loss_c += self.cls_criterion(torch.log(outputs[i][2]), targets[i][2])

            if opt.loss_box2d in ['sl1c']:
                self.box2d_criterion.set_mask(c_mask_mat[:, :, c_mask_i])
            if opt.loss_box2d is not None:
                if (i + opt.step_start) % (3 // opt.xyz) == 0:
                    loss_box2d += self.box2d_criterion(outputs[i][3], targets[i][3])

        losses = (loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d, loss_r_theta, loss_r_axis)
        return losses

    def compute_loss_set(self, outputs, targets, c_mask_mat, length, losses):
        eps = 0
        if 'img_all' in opt.node:
            eps = 1e-10
        loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d, loss_r_theta, loss_r_axis = losses
        st_criterion = nn.L1Loss().cuda()
        rot_criterion = nn.L1Loss().cuda()
        cls_criterion = nn.NLLLoss().cuda()
        box2d_criterion = nn.SmoothL1Loss().cuda()
        pred, gt = [], []
        for i in range(len(outputs)):
            pred.append(outputs[i][0])
        for i in range(len(targets)):
            gt.append(targets[i][0])
        batch_size = pred[0].size(0)
        loss_all = [torch.zeros(1)[0].cuda(), torch.zeros(1)[0].cuda(),
                    torch.zeros(1)[0].cuda(), torch.zeros(1)[0].cuda()]
        # print('*' * 100)
        # print('length', length)
        # print('c_mask_mat', c_mask_mat, torch.sum(c_mask_mat))
        # print(len(outputs))
        # print(len(targets))
        for b in range(batch_size):
            gt_length = torch.sum(c_mask_mat[b]).int().item()
            if opt.len_source != 'None':
                res_length = length[b, 0].int().item()
            else:
                res_length = gt_length
            if opt.reg_init == 'None':
                gt_length -= 1
                res_length -= 1
            assert gt_length <= len(targets) and res_length <= len(outputs)
            res_id = list(range(res_length))
            pred_id = []
            match_round = 1
            if opt.bi_match == 1:
                match_round = 2
            for m in range(match_round):
                if len(res_id) == 0:
                    print('res_length, gt_length', res_length, gt_length)
                    break
                gt_id = list(range(gt_length))
                if opt.bi_match == 2:
                    gt_id = gt_id * 2
                dis = np.zeros((len(res_id), len(gt_id)))
                for i in range(len(res_id)):#range(res_length):
                    for j in range(len(gt_id)):
                        ii = res_id[i]
                        jj = gt_id[j]
                        dis[i, j] = st_criterion(pred[ii][b:b + 1], gt[jj][b:b + 1]).item()
                # pdb.set_trace()
                for i in range(res_length):##max length, not critical
                    dis_min_id = np.argmin(dis)
                    res_i_i = dis_min_id // dis.shape[1]
                    gt_i_i = dis_min_id % dis.shape[1]
                    res_i = res_id.pop(res_i_i)
                    if not opt.one2many:
                        gt_i = gt_id.pop(gt_i_i)
                    else:
                        gt_i = gt_id[gt_i_i]
                    pred_id.append(res_i)
                    loss_all[0] += st_criterion(outputs[res_i][0][b:b+1], targets[gt_i][0][b:b+1])
                    loss_all[1] += rot_criterion(outputs[res_i][1][b:b+1], targets[gt_i][1][b:b+1])
                    # pdb.set_trace()
                    if opt.loss_c is not None:
                        loss_all[2] += cls_criterion(torch.log(outputs[res_i][2][b:b+1]+eps), targets[gt_i][2][b:b+1])
                    elif opt.len_adjust:
                        loss_all[2] += cls_criterion(torch.log(outputs[res_i][2][b:b + 1] + eps),
                                                     torch.Tensor([0]).long().cuda())
                    if opt.loss_box2d is not None:
                        loss_all[3] += box2d_criterion(outputs[res_i][3][b:b+1], targets[gt_i][3][b:b+1])
                    dis = np.delete(dis, res_i_i, axis=0)
                    if not opt.one2many:
                        dis = np.delete(dis, gt_i_i, axis=1)
                    if dis.shape[0] == 0 or dis.shape[1] == 0:
                        break
            # pdb.set_trace()
            if opt.len_adjust:  # inexistent part loss
                if opt.nms_thresh > 0:
                    # pdb.set_trace()##match redundant pred to regress, no more than two bg cls or dis thresh
                    res_id = self.remove_redundant_pred(pred_id, res_id, outputs)
                for res_i in res_id:
                    loss_all[2] += cls_criterion(torch.log(outputs[res_i][2][b:b+1]+eps),
                                                 torch.Tensor([opt.n_sem]).long().cuda()) * opt.cc_weight
        loss_x += loss_all[0] / batch_size
        loss_r += loss_all[1] / batch_size
        loss_c += loss_all[2] / batch_size
        loss_box2d += loss_all[3] / batch_size
        losses = (loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d, loss_r_theta, loss_r_axis)
        return losses

    def remove_redundant_pred(self, pred_id, res_id, outputs):
        bg_id = copy.deepcopy(res_id)
        for res_i in res_id:
            st, rot = outputs[res_i][0].detach(), outputs[res_i][1].detach()
            points = self.str_to_points(copy.deepcopy(st), copy.deepcopy(rot))
            for p in pred_id:
                st_p, rot_p = outputs[p][0].detach(), outputs[p][1].detach()
                points_p = self.str_to_points(copy.deepcopy(st_p), copy.deepcopy(rot_p))
                diag_1 = np.linalg.norm(points[0][6] - points[0][0])
                diag_2 = np.linalg.norm(points_p[0][6] - points_p[0][0])
                diag = max(diag_1, diag_2)
                dis = directed_hausdorff(points[0], points_p[0])[0] / diag  # 8 points of both
                if dis < opt.nms_thresh:
                    bg_id.remove(res_i)
                    break
        return bg_id

    def str_to_points(self, st, rot):
        prim_all = []
        st = st.cpu().numpy()
        rot = rot.cpu().numpy()
        s = st[0, 0:3] * opt.mean_std['mean_x'][0, 0] + opt.mean_std['std_x'][0, 0]
        t = st[0, 3:6] * opt.mean_std['mean_y'][0, 0] + opt.mean_std['std_y'][0, 0]
        r = rot[0] * opt.mean_std['mean_r'][0, 0] + opt.mean_std['std_r'][0, 0]
        prim_r = np.concatenate((s, t, r), axis=0)
        euler = prim_r[6:9]
        axis, theta = rot_euler_to_axis_theta(euler)
        prim_r = np.concatenate((np.zeros([10]), prim_r[0:6], axis, [theta]), axis=0)
        prim_all.append(prim_r)
        res_point = prim_all_to_cornerset(np.array(prim_all))
        return res_point

    def compute_loss_final(self, outputs, targets, c_mask_mat, length, losses):
        outputs, res_gt_id_all = outputs
        loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d, loss_r_theta, loss_r_axis = losses
        st_criterion = nn.L1Loss().cuda()
        rot_criterion = nn.L1Loss().cuda()
        batch_size = outputs[0][0].size(0)
        loss_all = [torch.zeros(1)[0].cuda(), torch.zeros(1)[0].cuda()]
        for b in range(batch_size):
            gt_length = torch.sum(c_mask_mat[b]).int().item()
            if opt.len_source != 'None':
                res_length = length[b, 0].int().item()
            else:
                res_length = gt_length
            # print(res_length, gt_length)
            if opt.reg_init == 'None':
                gt_length -= 1
                res_length -= 1
            assert gt_length <= len(targets) and res_length <= len(outputs)
            for i in range(res_length):
                res_i = i
                gt_i = res_gt_id_all[b][i]
                if gt_i < 0:
                    continue
                loss_all[0] += st_criterion(outputs[res_i][0][b:b+1], targets[gt_i][0][b:b+1])
                loss_all[1] += rot_criterion(outputs[res_i][1][b:b+1], targets[gt_i][1][b:b+1])
        loss_x += loss_all[0] / batch_size
        loss_r += loss_all[1] / batch_size
        losses = (loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d, loss_r_theta, loss_r_axis)
        return losses

    def compute_loss_batch(self, outputs, targets, c_mask_mat, length):
        # print(length, len(outputs[0]), len(outputs[1][0]), len(targets))
        # import pdb;pdb.set_trace()
        # loss_x, loss_r = Variable(torch.zeros(1)[0]), Variable(torch.zeros(1)[0])
        loss_x, loss_r, loss_c = torch.zeros(1)[0], torch.zeros(1)[0], torch.zeros(1)[0]
        loss_b, loss_e, loss_s = torch.zeros(1)[0], torch.zeros(1)[0], torch.zeros(1)[0]
        loss_box2d = torch.zeros(1)[0]
        loss_r_theta, loss_r_axis = torch.zeros(1)[0], torch.zeros(1)[0]
        if opt.use_gpu:
            loss_x, loss_r, loss_c = loss_x.cuda(), loss_r.cuda(), loss_c.cuda()
            loss_b, loss_e, loss_s = loss_b.cuda(), loss_e.cuda(), loss_s.cuda()
            loss_box2d = loss_box2d.cuda()
            loss_r_theta, loss_r_axis = loss_r_theta.cuda(), loss_r_axis.cuda()

        losses = (loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d, loss_r_theta, loss_r_axis)
        if opt.match == 'sequence':
            losses = self.compute_loss_sequence(outputs, targets, c_mask_mat, losses)
        elif opt.match == 'set':
            for i in range(opt.refine_mp):
                losses = self.compute_loss_set(outputs[i], targets, c_mask_mat, length, losses)
            if opt.sem_final:
                losses = self.compute_loss_final(outputs[opt.refine_mp], targets, c_mask_mat, length, losses)
        loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d, loss_r_theta, loss_r_axis = losses

        loss_x *= opt.gmm_weight
        if opt.out_r == 'class':
            loss_r_theta *= opt.r_weight
            loss_r_axis *= opt.r_weight
            loss_r = (loss_r_theta, loss_r_axis)
            loss = loss_x + loss_r_theta + loss_r_axis
        else:
            loss_r *= opt.r_weight
            loss = loss_x + loss_r
        if opt.loss_c is not None or opt.len_adjust:
            loss_c *= opt.c_weight
            loss += loss_c
        if opt.bbox_loss:
            loss += loss_b
        if opt.loss_e is not None:
            loss += loss_e
        if opt.loss_box2d is not None:
            loss_box2d *= opt.box2d_weight
            loss += loss_box2d
        if opt.sigma_reg > 0.:
            loss += loss_s
        return loss, loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d

    def compute_loss(self, outputs, targets, c_mask_mat, bbox, existence, prim_box_2d, length, phase=None):
        if opt.bi_lstm != None:
            # pass
            length = copy.deepcopy(length) * 2
        if opt.proposal == 'load' or opt.bi_lstm == 'online':
            losses = [torch.zeros(1)[0].cuda() for _ in range(8)]
            for b in range(len(outputs)):
                output, target = outputs[b], targets[b]
                losses_b = self.compute_loss_batch(output, target, c_mask_mat[b:b+1], length[b:b+1])
                for i in range(8):
                    losses[i] += losses_b[i]
            losses = tuple(x / len(outputs) for x in losses)
            return losses
        else:
            return self.compute_loss_batch(outputs, targets, c_mask_mat, length)

    def train(self):
        statis = StatisticsPrnn()
        if int(opt.GPU) == -1:
            opt.n_epochs = 800
        num_epochs = opt.n_epochs + 1
        if opt.proposal == 'save':
            num_epochs = 1
        if opt.proposal == 'load':
            data_sizes = {x: len(self.datasets[x]) for x in self.phases}
            self.model.load_proposals_all(data_sizes)
        for epoch in range(num_epochs):
            if epoch == 0 and opt.lr_r != opt.lr:
                self.update_optimizer()
            statis.print_epoch_begin(epoch)
            # self.scheduler.step()
            # print(optimizer.param_groups[0]['lr'])
            for phase in self.phases:
                if opt.fix_bn:
                    self.model.train(False)
                else:
                    if phase == 'train' or phase == 'train_val':
                        self.model.train(True)
                        # self.model.rgb_encoder.eval()
                        # self.model.rgb_encoder.fc_reg.train()
                        # self.model.rgb_encoder.fc_exist.train()
                    else:
                        self.model.train(False)
                statis.reset_epoch(phase)
                i_batch = -1
                for i_batch, data in enumerate(self.data_loaders[phase]):
                    input_mat, rot_mat, cls_mat, c_mask_mat, depth_mat, \
                    bbox, existence, prim_box_2d, length, item_id, box_proposal, box_gt_idx = data
                    # print('input1', type(depth_mat[0,0, 0, 0]), torch.max(depth_mat), depth_mat.size())
                    input_mat = input_mat.float()
                    c_mask_mat, depth_mat = c_mask_mat.float(), depth_mat.float()
                    bbox, existence = bbox.float(), existence.float()
                    prim_box_2d, length, item_id = prim_box_2d.float(), length.int(), item_id.int()
                    box_proposal, box_gt_idx = box_proposal.float(), box_gt_idx.int()
                    if opt.use_gpu:
                        input_mat, rot_mat, cls_mat = input_mat.cuda(), rot_mat.cuda(), cls_mat.cuda()
                        c_mask_mat, depth_mat = c_mask_mat.cuda(), depth_mat.cuda()
                        bbox, existence = bbox.cuda(), existence.cuda()
                        prim_box_2d, length, item_id = prim_box_2d.cuda(), length.cuda(), item_id.cuda()
                        box_proposal, box_gt_idx = box_proposal.cuda(), box_gt_idx.cuda()
                    # print('input2', type(depth_mat[0, 0, 0, 0]), torch.max(depth_mat), depth_mat.size())
                    # import pdb;pdb.set_trace()
                    if opt.out_r == 'theta':
                        rot_mat = rot_mat[:, :1*opt.xyz, :]
                    if opt.zero_in:
                        depth_mat = torch.ones(depth_mat.size()).cuda() * 0.
                    if opt.loss_e is None:
                        input_mat = input_mat[:, :2*opt.xyz, :]
                    if opt.gt_perturb > 0.:
                        input_mat[:, :2*opt.xyz, :] += (random.random() - 0.5) * opt.gt_perturb
                        rot_mat[:, :1*opt.xyz, :] += (random.random() - 0.5) * opt.gt_perturb
                    if 'proposal' in opt.faster:
                        outputs, targets = self.model(input_mat, rot_mat, cls_mat, depth_mat, c_mask_mat,
                                                      prim_box_2d, box_proposal, box_gt_idx, phase)
                        if opt.match == 'set':
                            targets, _ = targets
                        else:
                            targets, c_mask_mat = targets
                    else:
                        outputs, targets = self.model(input_mat, rot_mat, cls_mat, depth_mat, bbox, existence,
                                                      prim_box_2d, length, item_id, epoch, c_mask_mat, phase)
                    loss, loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d = self.compute_loss(
                        outputs, targets, c_mask_mat, bbox, existence, prim_box_2d, length)
                    if opt.gpu_base:
                        import pdb;pdb.set_trace()
                    if (phase == 'train' or phase == 'train_val') and epoch > 0:
                        if len(targets) != 0:
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                        else:
                            print(loss, loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d)
                    statis.accumulate(loss, loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d)
                    statis.print_batch(epoch, i_batch)
                is_best_tag = statis.is_best(i_batch, phase)
                if is_best_tag:
                    best_model = copy.deepcopy(self.model)
                    torch.save(best_model.cpu().state_dict(), '{}/best_model.pth'.format(opt.exp_dir))
                    print('=' * 40, 'update best model')
                statis.print_epoch_end(epoch, phase)
                if opt.visdom:
                    statis.plot(epoch, phase, is_best_tag)
            if epoch % opt.snapshot == 0:
                if opt.metric != '':
                    self.test_with_train(epoch, copy.deepcopy(self.model))
                snapshot = copy.deepcopy(self.model)
                torch.save(snapshot.cpu().state_dict(), '{}/model_epoch_{}.pth'.format(opt.exp_dir, epoch))
        statis.print_final()
        return

    def test_with_train(self, epoch, model):
        test_curve_one_epoch_init(epoch)
        opt.save_w_vector = False
        opt.demo = 'all'
        self.test(model=model, with_train=True)
        test_curve_one_epoch(epoch)
        opt.demo = None

    def test(self, model=None, with_train=False):
        if opt.demo == 'all':
            phases = ['test_'+x for x in opt.file_names['obj_classes']]
            # phases = ['test_chair', 'test_table', 'test_night_stand']
        else:
            phases = ['test_{}'.format(opt.demo)]
        for phase in phases:
            if with_train:
                data_loaders, data_sizes, datasets = self.build_dataloader_prnn_test()
                self.test_one_class(phase, data_sizes, data_loaders, datasets, model)
            else:
                if opt.proposal == 'load':
                    # pdb.set_trace()#####tmp
                    data_sizes = {'val': len(self.datasets[x]) for x in self.phases}
                    self.model.load_proposals_all(data_sizes)
                self.test_one_class(phase, self.data_sizes, self.data_loaders, self.datasets, self.model)

    def test_one_class(self, phase, data_sizes, data_loaders, datasets, model):
        if opt.ss_l2 is not None:
            if opt.test_version == 'train':
                file_name = 'length_{}_train.mat'.format(opt.ss_l2)
            # elif opt.test_version in ['1', 'ts']:
            elif 'ts' in opt.test_version or opt.test_version == '1':
                file_name = 'length_{}_test_{}.mat'.format(opt.ss_l2, opt.obj_class)
            else:
                file_name = 'length_{}_val.mat'.format(opt.ss_l2)
            print(file_name, '*'*30, 'stop_sign from resnet regression')
            # length_pred = scipy.io.loadmat('expssign/exp_0528_0000/{}'.format(file_name))
            len_dir = opt.len_dir
            root_dir = self.root_dir
            if opt.agnostic:
                if opt.obj_class[:6] == '3dprnn':
                    len_dir = '3dprnnall/' + len_dir + '_' + opt.obj_class
                else:
                    len_dir = 'all/' + len_dir + '_' + opt.obj_class
                root_dir = root_dir[:len(root_dir) - len(opt.obj_class)]
            file_name = os.path.join(root_dir, len_dir, file_name)
            length_pred = scipy.io.loadmat(file_name)
            length_pred = length_pred['length_{}'.format(opt.ss_l2)]
            # if opt.obj_class == 'table':
            if opt.extra_len > 0:
                length_pred = length_pred * (length_pred > 0) + opt.extra_len
            if opt.len_max > 0:
                length_pred = np.ones(length_pred.shape) * opt.len_max
        else:
            length_pred = None
        n_sem = opt.n_sem + int(opt.len_adjust)
        rs_res = torch.zeros(4 * data_sizes[phase], opt.max_len)
        cls_res = - torch.ones(data_sizes[phase], opt.max_len).long()
        invalid_res = - torch.ones(data_sizes[phase], opt.max_len).long()
        box2d_res = torch.zeros(4 * data_sizes[phase], opt.max_len)
        cls_prob_res = torch.zeros(n_sem * data_sizes[phase], opt.max_len)
        rot_prob_res = torch.zeros(31 * data_sizes[phase], opt.max_len)
        model.train(False)
        i_batch = -1
        running_loss = 0
        running_loss_x = 0
        running_loss_r = 0
        running_loss_c = 0
        running_loss_box2d = 0
        for i_batch, data in enumerate(data_loaders[phase]):
            if i_batch % opt.test_gap != 0:
                continue
            input_mat, rot_mat, cls_mat, c_mask_mat, depth_mat, \
            bbox, existence, prim_box_2d, length, item_id, box_proposal, box_gt_idx = data
            input_mat, rot_mat = input_mat.float(), rot_mat.float()
            c_mask_mat, depth_mat = c_mask_mat.float(), depth_mat.float()
            bbox, existence = bbox.float(), existence.float()
            prim_box_2d, length, item_id = prim_box_2d.float(), length.int(), item_id.int()
            box_proposal, box_gt_idx = box_proposal.float(), box_gt_idx.int()
            if opt.use_gpu:
                input_mat, rot_mat, cls_mat = input_mat.cuda(), rot_mat.cuda(), cls_mat.cuda()
                c_mask_mat, depth_mat = c_mask_mat.cuda(), depth_mat.cuda()
                bbox, existence = bbox.cuda(), existence.cuda()
                prim_box_2d, length, item_id = prim_box_2d.cuda(), length.cuda(), item_id.cuda()
                box_proposal, box_gt_idx = box_proposal.cuda(), box_gt_idx.cuda()
            if opt.out_r == 'theta':
                rot_mat = rot_mat[:, :1*opt.xyz]
            if opt.zero_in:
                depth_mat = torch.ones(depth_mat.size()).cuda() * 0.
            if opt.loss_e is None:
                input_mat = input_mat[:, :2*opt.xyz]
            if opt.ss_l2 is not None:
                length_i = length_pred[i_batch, 0]
            else:
                length_i = None
            if 'proposal' in opt.faster:
                outputs, targets = model(input_mat, rot_mat, cls_mat, depth_mat, c_mask_mat,
                                         prim_box_2d, box_proposal, box_gt_idx, 'test')
                targets, c_mask_mat = targets
                length_i = float(torch.sum(box_gt_idx>=0).item())
            else:
                outputs, targets = model(input_mat, rot_mat, cls_mat, depth_mat,
                                         bbox, existence, prim_box_2d, length, item_id,
                                         length_i=length_i)
            if opt.test_loss is not None:
                losses = self.compute_loss(outputs[1:]+[None], targets[:-1],
                                           c_mask_mat, bbox, existence, prim_box_2d, length, phase='val')
                loss, loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d = losses
                running_loss += loss.item()
                running_loss_x += loss_x.item()
                running_loss_r += loss_r.item()
                running_loss_c += loss_c.item()
                running_loss_box2d += loss_box2d.item()
                # print('LOSS    ', i_batch, running_loss / (i_batch + 1), losses)
                # if opt.test_batch > 1:
                continue
            if opt.filter:
                rs, cls, invalid, box2d = self.outputs_to_results_filter(outputs, phase, datasets, bbox, existence)
                cls_prob, rot_prob = 0, 0
            else:
                rs, cls, invalid, box2d, cls_prob, rot_prob = self.outputs_to_results(outputs, phase, datasets,
                                                            bbox, existence, length=length_i, i_batch=i_batch)
            if opt.global_denorm:
                rs = self.denormalize_prims_wrt_one_prim(rs, i_batch)
            rs_res[i_batch * 4: i_batch * 4 + 4, :] = rs
            cls_res[i_batch, :] = cls
            invalid_res[i_batch, :] = invalid
            box2d_res[i_batch * 4: i_batch * 4 + 4, :] = box2d
            cls_prob_res[i_batch * n_sem: i_batch * n_sem + n_sem, :] = cls_prob
            rot_prob_res[i_batch * 31: i_batch * 31 + 31, :] = rot_prob
            res = {'x': rs_res.numpy(), 'cls': cls_res.numpy(), 'invalid': invalid_res.numpy(),
                   'box2d': box2d_res.numpy(), 'cls_prob': cls_prob_res.numpy(),
                   'rot_prob': rot_prob_res.numpy()}
            if opt.encoder is not None:
                scipy.io.savemat('{}/test_res_mn_{}.mat'.format(opt.init_source, phase[5:]), res)
            else:
                scipy.io.savemat('{}/test_res_mn_pure.mat'.format(opt.init_source), res)
        return (running_loss / (i_batch + 1), running_loss_x / (i_batch + 1), running_loss_r / (i_batch + 1),
                running_loss_c / (i_batch + 1), running_loss_box2d / (i_batch + 1))

    def denorm_test(self):
        dir = opt.exp_prefix + '/' + opt.env + '_v{}_{}'.format('2', opt.model_epoch)
        res = scipy.io.loadmat('{}/test_res_mn_{}.mat'.format(dir, opt.obj_class))
        for i in range(res['x'].shape[0] // 4):
            rs = copy.deepcopy(res['x'][i * 4:i * 4 + 4, :])
            rs = self.denormalize_prims_wrt_one_prim(rs, i)
            res['x'][i * 4:i * 4 + 4, :] = rs
        scipy.io.savemat('{}/test_res_mn_{}.mat'.format(opt.init_source, opt.obj_class), res)

    def denormalize_prims_wrt_one_prim(self, res_prim, i_batch):
        # print('before 0', res_prim)
        scale, trans = self.get_obj_scale_translation_all(copy.deepcopy(res_prim), i_batch)
        # print('after', res_prim)
        res_prim[0, :] *= scale
        res_prim[1, :] *= scale
        stop_idx = [ind for (ind, x) in enumerate(res_prim[0, :]) if x == 0][0]
        for col in range(0, stop_idx - 2, 3):
            res_prim[1, col] += trans[0]
            res_prim[1, col + 1] += trans[1]
            res_prim[1, col + 2] += trans[2]
        return res_prim

    def get_obj_scale_translation_all(self, res_prim, i):
        voxel_scale = opt.v_size
        obj_center = np.array([voxel_scale / 2, voxel_scale / 2, voxel_scale / 2])
        res_prim = res_obj_to_gt_prims(i, res_prim)  # 2844,100 matrix
        res_prim = np.array(res_prim)
        # print('before 1', res_prim)
        res_point = prim_all_to_cornerset(res_prim)
        # print('res_point', res_point)
        if res_point == []:
            print('No primitives for test instance: ', i)
            return 1, np.zeros(3)
        res_point = np.concatenate(tuple(res_point), axis=0)
        min_xyz = np.min(res_point, axis=0)
        max_xyz = np.max(res_point, axis=0)
        # print('min', min_xyz)
        # print('max', max_xyz)
        len_xyz = max_xyz - min_xyz
        # print('len', len_xyz)
        scale = voxel_scale / len_xyz
        # print('scale', scale)
        scale = np.min(scale)
        # print('scale', scale)
        min_xyz *= scale
        max_xyz *= scale
        c_xyz = min_xyz + (max_xyz - min_xyz) / 2
        # print('c_xyz', c_xyz)
        trans = obj_center - c_xyz
        # print('trans', trans)
        return scale, trans

    def rot_to_theta_axis(self, r):
        theta = r[0, :27]
        axis = r[0, 27:]

    def r_to_rot(self, r, j):
        if opt.out_r == 'class':
            rot = r
        elif opt.out_r == '3dprnn':
            rot = torch.cat((r[:, j:j + 1], r[:, j + 3:j + 4]), dim=1)
        else:
            rot = r[:, j:j + 1]
        return rot

    def outputs_xyz_three_steps(self, outputs):
        new_outs = []
        for i in range(len(outputs)):
            if len(outputs[i]) == 4:
                x, r, c, b = outputs[i]
                for j in range(3):
                    rot = self.r_to_rot(r, j)
                    new_outs.append((torch.cat((x[:, j:j + 1], x[:, j + opt.xyz:j + 1 + opt.xyz]), dim=1),
                                     rot, c, b))
            else:
                x, r, c, b, y = outputs[i]
                for j in range(3):
                    rot = self.r_to_rot(r, j)
                    new_outs.append((torch.cat((x[:, j:j + 1], x[:, j + opt.xyz:j + 1 + opt.xyz]), dim=1),
                                     rot, c, b, y))
        return new_outs

    def drop_outputs(self, outputs, targets, length):
        outputs_a, outputs_b = [], []
        dis_paired = []
        st_criterion = nn.L1Loss().cuda()
        pred, gt = [], []
        for i in range(len(outputs)):
            pred.append(outputs[i][0])
        for i in range(len(targets)):
            gt.append(targets[i][0])
        batch_size = pred[0].size(0)
        assert batch_size == 1
        for b in range(batch_size):
            assert length == len(targets) and length == len(outputs)
            res_id = list(range(length))
            gt_id = list(range(length))
            dis = np.zeros((length, length))
            for i in range(length):
                for j in range(length):
                    if i <= j:
                        dis[i, j] = 1000
                    else:
                        dis[i, j] = st_criterion(pred[i][b:b + 1], gt[j][b:b + 1]).item()
            # import pdb;pdb.set_trace()
            for i in range(length):
                dis_min = np.min(dis)
                dis_min_id = np.argmin(dis)
                res_i_i = dis_min_id // dis.shape[1]
                gt_i_i = dis_min_id % dis.shape[1]
                res_i = res_id.pop(res_i_i)
                gt_i = gt_id.pop(gt_i_i)
                res_id.remove(gt_i)
                gt_id.remove(res_i)
                dis = np.delete(dis, [res_i_i, gt_i_i], axis=0)
                dis = np.delete(dis, [res_i_i, gt_i_i], axis=1)
                res_cls = outputs[res_i][2][b:b + 1]
                gt_cls = targets[gt_i][2][b:b + 1]
                # pdb.set_trace()
                if opt.drop_test == 1.5:
                    if random.random() < 0.5:# or torch.argmax(gt_cls).item() == opt.n_sem:
                        outputs_a.append(outputs[res_i])
                        outputs_b.append(targets[gt_i])
                    else:
                        outputs_a.append(targets[gt_i])
                        outputs_b.append(outputs[res_i])
                else:
                    if torch.max(res_cls) > torch.max(gt_cls):# or torch.argmax(gt_cls).item() == opt.n_sem:
                        outputs_a.append(outputs[res_i])
                        outputs_b.append(targets[gt_i])
                    else:
                        outputs_a.append(targets[gt_i])
                        outputs_b.append(outputs[res_i])
                dis_paired.append(dis_min)
                if dis.shape[0] == 0 or dis.shape[1] == 0:
                    break
        # pdb.set_trace()
        if opt.drop_test >= 2:
            # pdb.set_trace()
            length /= 2
            r_id, t_std = 1, 1
            if opt.drop_test == 2:
                r_id = int(length * 0.8)
                t_std = 3
            if opt.drop_test == 3:
                r_id = int(length * 0.9)
                t_std = 10
            dis_mean = np.mean(dis_paired[:r_id])
            dis_std = np.std(dis_paired[:r_id])
            for i in range(len(dis_paired[r_id:])):
                d = dis_paired[r_id:][i]
                if abs(d - dis_mean) > t_std * dis_std:
                    outputs_a.append(outputs_b[i + r_id])
        return outputs_a

    def outputs_to_results(self, outputs, phase, datasets, bbox=None, existence=None, length=None, i_batch=None):
        rs = torch.zeros(4, opt.max_len)
        cls = - torch.ones(1, opt.max_len).long()
        box2d = torch.zeros(4, opt.max_len)
        cls_prob = torch.zeros(opt.n_sem + int(opt.len_adjust), opt.max_len)
        rot_prob = torch.zeros(31, opt.max_len)
        if length == 0:
            return rs, cls, -1, box2d, cls_prob, rot_prob
        seq_length = opt.pred_len
        if opt.bi_lstm != None:
            length = int(length) * 2
            seq_length = seq_length * 2
            assert length == len(outputs)
            if opt.bi_match != None and opt.drop_test != None:
                outputs = self.drop_outputs(copy.deepcopy(outputs), copy.deepcopy(outputs), length)
                length = len(outputs)
        if opt.xyz == 3:
            outputs = self.outputs_xyz_three_steps(outputs)
        if i_batch < 5:
            print(length)
        # assert length * 3 == len(outputs)
        mean_x, mean_y, mean_r, std_x, std_y, std_r = datasets[phase].get_mean_std()
        one_prim_flag = False
        for i in range(seq_length):
            # if opt.debug == 'denorm_label_train_to_test':
            if len(outputs[i]) == 4:
                x, r, c, b = outputs[i]
            else:
                x, r, c, b, y = outputs[i]
            # x, r, c, b, y = outputs[i]
            if not opt.global_denorm:
                x[:, 0] = x[:, 0] * std_x[0, 0] + mean_x[0, 0]
                x[:, 1] = x[:, 1] * std_y[0, 0] + mean_y[0, 0]
                if opt.out_r != 'class':
                    r[:, 0] = r[:, 0] * std_r[0, 0] + mean_r[0, 0]

            assert x.size()[0] == 1 and r.size()[0] == 1# and c.size()[0] == 1
            rs[0, i] = x[0, 0]
            rs[1, i] = x[0, 1]
            if opt.out_r != 'class':
                rs[2, i] = r[0, 0]
            if opt.out_r == '3dprnn':
                rs[3, i] = r[0, 1]
            # elif opt.out_r == 'theta':
            #     rs[3, i] = 0
            # print(rs[:, :i+2])
            if opt.loss_c is not None or opt.loss_box2d is not None or opt.len_adjust:
                if opt.init_in is None: # not examined
                    cls_i = i
                    cls_ii = i
                else:
                    import pdb;pdb.set_trace()
                    cls_i = i + 1
                    cls_ii = i - 2
                if cls_i % 3 == 0:
                    if opt.loss_c is not None or opt.len_adjust:
                        _, cls[0, cls_ii] = torch.max(c[0, :], 0)
                        # import pdb;pdb.set_trace()
                        if opt.reg_init == 'None' and c.size(1) < cls_prob.size(0):
                            c = torch.cat((c, torch.ones(1, 1)*1e-8), dim=1)
                        cls_prob[:, cls_ii] = c[0, :]
                    if opt.loss_box2d is not None:
                        box2d[:, cls_ii] = b[0, :]
                    if opt.out_r == 'class':
                        # rs[2, i:i + 3] = 0
                        # rs[3, i:i + 3] = 0
                        rot_prob[:, cls_ii] = r[0, :]
                        _, theta = torch.max(r[0, :27], 0)
                        _, axis = torch.max(r[0, 27:], 0)
                        if axis.item() > 0:
                            theta = (theta.float() - 13) * 6 / 100.
                            axis = axis.item() - 1
                            rs[2, i + axis] = theta
            if one_prim_flag:
                if i == 2:
                    break
                else:
                    continue
            else:
                if length is not None:
                    if i == length * 3 - 1:
                        # print('i, length', i, length)
                        break
                else:
                    if x[0, 2] == 1:
                        if i < 2:
                            one_prim_flag = True
                        else:
                            break
        return rs, cls, -1, box2d, cls_prob, rot_prob

    def denormalize_bbox(self, bbox, mean_std_init):
        # bbox *= float(opt.v_size)
        mean_x = mean_std_init['mean_x']
        mean_y = mean_std_init['mean_y']
        # mean_r = mean_std_init['mean_r']
        std_x = mean_std_init['std_x']
        std_y = mean_std_init['std_y']
        # std_r = mean_std_init['std_r']
        if not opt.global_denorm:
            for i_sem in range(opt.n_sem):
                for j in range(3):
                    #### wrong before, bbox is min and len, mean_x is for scale and mean_y is for translation
                    #### but it is ok cus normalize shares the same means and stds
                    # bbox[0, i_sem * opt.n_para + j] = bbox[0, i_sem * opt.n_para + j] * std_x[0, 0] +  mean_x[0, 0]
                    # bbox[0, i_sem * opt.n_para + j + 3] = bbox[0, i_sem * opt.n_para + j + 3] * std_y[0, 0] + mean_y[0, 0]
                    bbox[0, i_sem * opt.n_para + j] = bbox[0, i_sem * opt.n_para + j] * std_y[0, 0]\
                                                      + mean_y[0, 0]
                    bbox[0, i_sem * opt.n_para + j + 3] = bbox[0, i_sem * opt.n_para + j + 3] * std_x[0, 0]\
                                                          + mean_x[0, 0]
        bbox /= float(opt.v_size)
        return bbox

    def check_in_box(self, rs, cls, bbox, existence, mean_std_init):
        # print('bbox before denorm', bbox)
        bbox = self.denormalize_bbox(bbox.clone(), mean_std_init)
        # print('bbox after denorm', bbox)
        # import pdb
        # pdb.set_trace()
        in_bbox = [True, True, True]
        rs = rs.cpu().detach().numpy()
        theta = np.max(rs[2, :])
        axis = np.argmax(rs[2, :])
        rotation = [0, 0, 0, theta]
        rotation[axis] = 1
        prim = np.array([[0 for i in range(10)] + [rs[0, 0], rs[0, 1], rs[0, 2]] +
                         [rs[1, 0], rs[1, 1], rs[1, 2]] + rotation])
        point_set = prim_all_to_cornerset(prim)
        point_set = point_set[0]    # (8, 3)
        if not existence[0, cls]:
            return in_bbox
        sem_bbox = bbox[0, cls * opt.n_para : (cls + 1) * opt.n_para]
        # import pdb
        # pdb.set_trace()
        for i in range(8):
            for xyz in range(3):
                if point_set[i, xyz] < sem_bbox[xyz]-opt.f_thresh or point_set[i, xyz] > sem_bbox[xyz]+sem_bbox[xyz + 3]+opt.f_thresh:
                    in_bbox[xyz] = False
        return in_bbox

    def outputs_to_results_filter(self, outputs, phase, datasets, bbox=None, existence=None):
        ##### check the synchronization of x and y
        # import pdb
        # pdb.set_trace()
        # file_name = opt.file_names['mean_std'][opt.obj_class]
        # mean_std_init = scipy.io.loadmat(os.path.join(self.root_dir, file_name))
        mean_std_init = opt.mean_std
        rs = torch.zeros(4, opt.max_len)
        cls = - torch.ones(1, opt.max_len).long()
        invalid = - torch.ones(1, opt.max_len).long()
        box2d = torch.zeros(4, opt.max_len)
        mean_x, mean_y, mean_r, std_x, std_y, std_r = datasets[phase].get_mean_std()
        ending = False
        preservation = torch.zeros(4, 3)
        for prim_i in range(opt.pred_len // 3):
            in_bbox = [True, True, True]
            c_j = -1
            for c_j in range(opt.n_component):
                i = -1
                for xyz_t in range(3):
                    i = prim_i * 3 + xyz_t
                    x, r, c, b, y = outputs[i]
                    x = x.detach().clone()
                    r = r.detach().clone()
                    c = c.detach().clone()
                    b = b.detach().clone()
                    y = y.detach().clone()
                    if c_j > 0 and not in_bbox[xyz_t]:
                        x = self.model.get_x(y, i_component=c_j)
                    if not opt.global_denorm:
                        x[:, 0] = x[:, 0] * std_x[0, 0] + mean_x[0, 0]
                        x[:, 1] = x[:, 1] * std_y[0, 0] + mean_y[0, 0]
                        r[:, 0] = r[:, 0] * std_r[0, 0] + mean_r[0, 0]

                    assert x.size()[0] == 1 and r.size()[0] == 1# and c.size()[0] == 1
                    rs[0, i] = x[0, 0]
                    rs[1, i] = x[0, 1]
                    rs[2, i] = r[0, 0]
                    if opt.out_r == '3dprnn':
                        rs[3, i] = r[0, 1]
                    elif opt.out_r == 'theta':
                        rs[3, i] = 0
                    # if opt.loss_c is not None:
                    #     if i % 3 == 0:
                    #         _, cls[0, i] = torch.max(c[0, :], 0)
                    if opt.loss_c is not None or opt.loss_box2d is not None or opt.len_adjust:
                        if opt.init_in is None:  # not examined
                            cls_i = i
                            cls_ii = i
                        else:
                            cls_i = i + 1
                            cls_ii = i - 2
                        if cls_i % 3 == 0:
                            if opt.loss_c is not None or opt.len_adjust:
                                _, cls[0, cls_ii] = torch.max(c[0, :], 0)
                            if opt.loss_box2d is not None:
                                box2d[:, cls_ii] = b[0, :]
                    if x[0, 2] == 1:
                        ending = True
                        break
                if c_j == 0:
                    preservation = rs[:, i - 2:i + 1].clone()
                if opt.filter:
                    if (i + 1) % 3 == 0:
                        in_bbox = self.check_in_box(copy.deepcopy(rs[0:3, i - 2:i + 1]),
                                                    copy.deepcopy(cls[0, i - 2]),
                                                    bbox, existence, mean_std_init)
                        if in_bbox[0] and in_bbox[1] and in_bbox[2]:
                            break
                # if ending:
                #     break
            invalid[0, prim_i] = c_j
            if not (in_bbox[0] and in_bbox[1] and in_bbox[2]):
                invalid[0, prim_i] = opt.n_component
                rs[:, prim_i * 3:prim_i * 3 + 3] = preservation
            if ending:
                break

        return rs, cls, invalid, box2d

    def getVx(self, Rv):
        vx = np.array([[0, -Rv[2], Rv[1]],
                       [Rv[2], 0, -Rv[0]],
                       [-Rv[1], Rv[0], 0]])
        return vx

    def save_train_val_test_w(self):
        # phases = ['train', 'val', 'test_chair', 'test_table', 'test_night_stand']
        save_w_vectors = {x: torch.zeros(self.data_sizes[x], opt.con_size) for x in self.phases}
        save_box2d = {x: torch.zeros(self.data_sizes[x], 4) for x in self.phases}
        self.model.train(False)
        for phase in self.phases:
            i_batch = -1
            for i_batch, data in enumerate(self.data_loaders[phase]):
                depth_mat, prim_box_2d = data
                depth_mat, prim_box_2d = depth_mat.float(), prim_box_2d.float()
                if opt.use_gpu:
                    depth_mat, prim_box_2d = depth_mat.cuda(), prim_box_2d.cuda()
                outputs = self.model(depth_mat)
                if opt.fpn:
                    outputs = outputs[0]
                save_w_vectors[phase][i_batch, :] = outputs.detach().cpu()
                save_box2d[phase][i_batch, :] = prim_box_2d[0, :, 0]
                # print('save_train_val_test_w', i_batch)
                # print(prim_box_2d)
            w_vector_name = os.path.join(opt.init_source, 'w_vector_{}.mat'.format(phase))
            scipy.io.savemat(w_vector_name, {'x': save_w_vectors[phase].numpy(),
                                             'box2d': save_box2d[phase].numpy()})
            print(phase, 'done', i_batch)

    def val(self):
        statis = StatisticsPrnn()
        # if opt.save_nn:
        #     statis.init_save_nn(self.data_sizes, 'train_val')
        for epoch in range(0, opt.n_epochs + 1, opt.snapshot):
            statis.print_epoch_begin(epoch)
            # self.scheduler.step()
            # print(optimizer.param_groups[0]['lr'])
            model_dict = torch.load(os.path.join(opt.exp_prefix + '/exp_' + '0521_4',
                                                 'model_epoch_{}.pth'.format(epoch)))
            self.model.load_state_dict(model_dict)
            for phase in ['val']:
                if opt.fix_bn:
                    self.model.train(False)
                else:
                    if phase == 'train' or phase == 'train_val':
                        self.model.train(True)
                        # self.model.rgb_encoder.eval()
                        # self.model.rgb_encoder.fc_reg.train()
                        # self.model.rgb_encoder.fc_exist.train()
                    else:
                        self.model.train(False)
                statis.reset_epoch(phase)
                i_batch = -1
                for i_batch, data in enumerate(self.data_loaders[phase]):
                    input_mat, rot_mat, cls_mat, c_mask_mat, depth_mat, bbox, existence, prim_box_2d, length = data
                    input_mat, rot_mat = input_mat.float(), rot_mat.float()
                    c_mask_mat, depth_mat = c_mask_mat.float(), depth_mat.float()
                    bbox, existence = bbox.float(), existence.float()
                    prim_box_2d, length = prim_box_2d.float(), length.int()
                    if opt.use_gpu:
                        input_mat, rot_mat, cls_mat = input_mat.cuda(), rot_mat.cuda(), cls_mat.cuda()
                        c_mask_mat, depth_mat = c_mask_mat.cuda(), depth_mat.cuda()
                        bbox, existence = bbox.cuda(), existence.cuda()
                        prim_box_2d, length = prim_box_2d.cuda(), length.cuda()
                    if opt.out_r == 'theta':
                        rot_mat = rot_mat[:, 0:1, :]
                    if opt.zero_in:
                        depth_mat = torch.ones(depth_mat.size()).cuda() * 0.
                    if opt.gt_perturb > 0.:
                        input_mat[:, :2, :] += (random.random() - 0.5) * opt.gt_perturb
                        rot_mat[:, :1, :] += (random.random() - 0.5) * opt.gt_perturb
                    outputs, targets = self.model(input_mat, rot_mat, cls_mat, depth_mat,
                                                  bbox, existence, prim_box_2d, length, epoch)
                    loss, loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d = self.compute_loss(
                        outputs, targets, c_mask_mat, bbox, existence, prim_box_2d, length)
                    if (phase == 'train' or phase == 'train_val') and epoch > 0:# and not opt.save_nn:
                        if len(targets) != 0:
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                        else:
                            print(loss, loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d)
                    statis.accumulate(loss, loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d)
                    statis.print_batch(epoch, i_batch)
                is_best_tag = statis.is_best(i_batch, phase)
                if is_best_tag:# and not opt.save_nn:
                    best_model = copy.deepcopy(self.model)
                    # torch.save(best_model.cpu().state_dict(), '{}/best_model.pth'.format(opt.exp_dir))
                    print('=' * 40, 'update best model')
                statis.print_epoch_end(epoch, phase)
                if opt.visdom:
                    statis.plot(epoch, phase, is_best_tag)
            if epoch % opt.snapshot == 0:# and not opt.save_nn:
                snapshot = copy.deepcopy(self.model)
                # torch.save(snapshot.cpu().state_dict(), '{}/model_epoch_{}.pth'.format(opt.exp_dir, epoch))
        statis.print_final()
        return
