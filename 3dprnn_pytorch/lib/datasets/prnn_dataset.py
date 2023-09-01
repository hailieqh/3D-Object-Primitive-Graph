import cv2
import json
import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchfile
from torchvision import transforms
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')

from lib.opts import *
from lib.utils.prnn_utils import *

# global opt
# opt = get_opt()

class SYNSet(Dataset):
    def __init__(self, inverse, transform=None, phase='train'):
        super(SYNSet, self).__init__()
        global opt
        opt = get_opt()
        self.inverse = inverse
        self.transform = transform
        self.phase = phase
        self.root_dir = opt.data_dir
        intervals = opt.intervals
        if phase[:4] == 'test':
            self.obj_class = phase[5:]
        else:
            self.obj_class = opt.obj_class
        # file_name = opt.file_names['mean_std'][self.obj_class]
        # self.test_init = scipy.io.loadmat(os.path.join(self.root_dir, file_name))
        self.test_init = opt.mean_std
        if opt.test_loss is not None:
            self.test_init = load_test_init(self.obj_class)
        if self.phase == 'train_val' and opt.encoder == 'depth':
            labels_train = load_labels_from_t7('train', self.root_dir, intervals)
            labels_val = load_labels_from_t7('val', self.root_dir, intervals)
            self.labels = combine_labels(labels_train, labels_val, intervals['train'], intervals['val'])
            depth_train = load_depth('train', self.root_dir, intervals=intervals)
            depth_val = load_depth('val', self.root_dir, intervals=intervals)
            self.depth = combine_depth(depth_train, depth_val, intervals['train'], intervals['val'])
        else:
            if opt.encoder in ['resnet', 'hg', 'depth_new']:
                self.image_names = load_names(self.phase, self.root_dir, self.obj_class)
                self.length_all = load_length(self.phase, self.root_dir, self.image_names, self.obj_class)
                # self.labels = load_labels_from_t7(self.phase, self.root_dir, intervals)
                self.labels = load_primset_to_t7_format(self.root_dir, self.obj_class)
                self.match_id = load_model_id(self.root_dir, self.obj_class)
                if opt.bbox_con[:3] == 'ora':
                    self.bbox_3d = load_bbox_3d(opt.file_names['primset'], self.root_dir)
                    # file_name = opt.file_names['mean_std'][opt.obj_class]
                    # self.mean_std_init = scipy.io.loadmat(os.path.join(self.root_dir, file_name))
                    self.mean_std_init = opt.mean_std
                if opt.loss_box2d is not None or opt.box2d_source is not None:
                    self.prim_points_2d = load_prim_points_2d(self.root_dir, self.obj_class)
            else:
                self.labels = load_labels_from_t7(self.phase, self.root_dir, intervals)
                if opt.encoder is None:
                    self.labels = self.down_sample_labels()
                self.depth = load_depth(self.phase, self.root_dir, intervals=intervals)
                if opt.cut_data:
                    self.labels = self.labels[:700]
                    self.depth = self.depth[:700]

        self.init = None
        if opt.init_in is not None:
            self.init = get_init(self.labels)
            self.labels = labels_add_init(self.labels, self.init)
        self.max_length = get_max_length(self.labels, self.obj_class)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if 'proposal' in opt.faster:
            self.box_proposals = load_box_proposals(self.root_dir, self.obj_class)

    def __getitem__(self, item):
        ###############2d box -- normalize! 0-1 w h
        if 'all' in self.obj_class:
            return self.class_agnostic(item)
        bbox = np.zeros(opt.n_sem * opt.n_para)
        existence = np.zeros(opt.n_sem)
        prim_box_2d = torch.zeros(4, self.max_length)
        length = torch.zeros(1)
        box_proposal = torch.zeros(1)
        box_gt_idx = torch.zeros(1)
        if opt.encoder in ['resnet', 'hg', 'depth_new']:
            image_name = self.image_names[item]
            id_img_ori = int(image_name[0:4])   # 1-3839
            image = load_image(image_name, self.root_dir)
            length[0] = self.length_all[item, 0]
            # bbox_2d = copy.deepcopy(self.bbox_2d[:, id_img_ori - 1])
            # image = self.crop_bbox(image, bbox_2d)
            voxel_id = get_model_id(id_img_ori, self.match_id)    # 0-215
            if opt.bbox_con[:3] == 'ora':
                part, existence = get_bbox_3d_ins(self.bbox_3d[voxel_id, 0]['sem'][0, 0])
                bbox = normalize_bbox(part, self.mean_std_init)
            if opt.loss_box2d is not None or opt.box2d_source is not None:
                prim_box_2d, sample = get_prim_box_2d(item, self.match_id, id_img_ori, self.prim_points_2d,
                                                      self.max_length, image, self.transform, inverse=self.inverse)
            else:
                sample = transform_image_label(id_img_ori, image, transform=self.transform)
            input_mat, rot_mat, cls_mat, c_mask_mat, prim_box_2d = get_label(voxel_id, self.labels, self.max_length,
                                                                             self.init, self.test_init, item, prim_box_2d,
                                                                             inverse=self.inverse)
            if opt.encoder == 'resnet':
                assert torch.sum(sample['image'] > 1.) == 0 and torch.sum(sample['image'] < 0.) == 0
                sample['image'] = self.normalize(sample['image'])
            if 'proposal' in opt.faster:
                box_proposal = self.box_proposals[image_name]
                box_proposal = box_proposal_nms(box_proposal)
                num_prop = len(box_proposal)
                if num_prop == 0:
                    h, w, _ = image.shape
                    box_proposal = [[w*0.1, h*0.1, w*0.9, h*0.9]]
                    num_prop = len(box_proposal)
                box_proposal = box_proposal_transform(box_proposal, id_img_ori, opt.n_prop, image, self.transform)
                box_gt_idx = match_2d_box(box_proposal, prim_box_2d, num_prop, id=image_name)# numpy automatically to tensor?
            return input_mat, rot_mat, cls_mat, c_mask_mat, sample['image'], \
                   bbox, existence, prim_box_2d, length, torch.Tensor([item]), box_proposal, box_gt_idx
        else:
            input_mat, rot_mat, cls_mat, c_mask_mat, _ = get_label(item, self.labels, self.max_length,
                                                                   inverse=self.inverse)
            depth_mat = copy.deepcopy(self.depth[item, :, :])   # (64, 64)
            depth_mat = torch.from_numpy(depth_mat[np.newaxis, :, :])  # (1, 64, 64)
            return input_mat, rot_mat, cls_mat, c_mask_mat, depth_mat, \
                   bbox, existence, prim_box_2d, length, torch.Tensor([item]), box_proposal, box_gt_idx

    def __len__(self):
        if opt.encoder in ['resnet', 'hg', 'depth_new']:
            if 'all' in self.obj_class:
                length = 0
                for cls in opt.file_names['obj_classes']:
                    length += len(self.image_names[cls])
                return length
            return len(self.image_names)
        else:
            return len(self.labels)

    def class_agnostic(self, item):
        item, cls = get_obj_class(item, self.phase)
        bbox = np.zeros(opt.n_sem * opt.n_para)
        existence = np.zeros(opt.n_sem)
        prim_box_2d = torch.zeros(4, self.max_length)
        length = torch.zeros(1)
        if opt.encoder in ['resnet', 'hg', 'depth_new']:
            image_name = self.image_names[cls][item]
            id_img_ori = int(image_name[0:4])  # 1-3839
            image = load_image(image_name, os.path.join(self.root_dir, cls))
            length[0] = self.length_all[cls][item, 0]##***
            # bbox_2d = copy.deepcopy(self.bbox_2d[:, id_img_ori - 1])
            # image = self.crop_bbox(image, bbox_2d)
            voxel_id = get_model_id(id_img_ori, self.match_id[cls])  # 0-215
            if opt.loss_box2d is not None or opt.box2d_source is not None:
                prim_box_2d, sample = get_prim_box_2d(item, self.match_id[cls], id_img_ori, self.prim_points_2d[cls],
                                                      self.max_length, image, self.transform, inverse=self.inverse)
            else:
                sample = transform_image_label(id_img_ori, image, transform=self.transform)
            input_mat, rot_mat, cls_mat, c_mask_mat, prim_box_2d = get_label(voxel_id, self.labels[cls], self.max_length,
                                                                             self.init, self.test_init[cls], item,
                                                                             prim_box_2d,
                                                                             inverse=self.inverse)
            if opt.encoder == 'resnet':
                assert torch.sum(sample['image'] > 1.) == 0 and torch.sum(sample['image'] < 0.) == 0
                sample['image'] = self.normalize(sample['image'])
            return input_mat, rot_mat, cls_mat, c_mask_mat, sample['image'], \
                   bbox, existence, prim_box_2d, length, torch.Tensor([item]), torch.Tensor([item]), torch.Tensor([item])

    def down_sample_labels(self):
        labels_sampled = []
        for i in range(0, len(self.labels), 5):
            labels_sampled.append(self.labels[i])
        return labels_sampled

    def get_mean_std(self):
        return (self.test_init['mean_x'], self.test_init['mean_y'], self.test_init['mean_r'],
                self.test_init['std_x'], self.test_init['std_y'], self.test_init['std_r'])

    def get_test_id(self):
        test_id = []
        for item in range(len(self.image_names)):
            image_name = self.image_names[item]
            id_img_ori = int(image_name[0:4])
            voxel_id = get_model_id(id_img_ori, self.match_id)
            test_id.append(voxel_id)
        return test_id


class SYNTestSet(Dataset):
    def __init__(self, inverse, transform=None, phase='train'):
        super(SYNTestSet, self).__init__()
        global opt
        opt = get_opt()
        self.inverse = inverse
        self.transform = transform
        self.phase = phase
        self.root_dir = opt.data_dir
        if phase[:4] == 'test':
            self.obj_class = phase[5:]
        else:
            self.obj_class = opt.obj_class
        # self.path = '/Users/heqian/Research/projects/primitive-based_3d/data/3dprnn'
        if opt.reg_init != 'None':
            self.test_init = opt.mean_std
        else:
            self.test_init = load_test_init(self.obj_class)
        intervals = opt.intervals
        if opt.encoder in ['resnet', 'hg', 'depth_new']:
            self.image_names = load_names(self.phase, self.root_dir, self.obj_class)
            self.length_all = load_length(self.phase, self.root_dir, self.image_names, self.obj_class)
            # self.bbox_2d = self.load_bbox_2d('bboxes.mat')
            # self.labels = load_labels_from_t7(self.phase, self.root_dir, intervals)
            self.labels = load_primset_to_t7_format(self.root_dir, self.obj_class)
            self.match_id = load_model_id(self.root_dir, self.obj_class)
            if opt.bbox_con[:3] == 'ora' or opt.filter:
                self.match_id = load_model_id(self.root_dir, self.obj_class)
                self.bbox_3d = load_bbox_3d(opt.file_names['primset'], self.root_dir)
                # file_name = opt.file_names['mean_std'][opt.obj_class]
                # self.mean_std_init = scipy.io.loadmat(os.path.join(self.root_dir, file_name))
                self.mean_std_init = opt.mean_std
            if opt.loss_box2d is not None or opt.box2d_source is not None:
                self.prim_points_2d = load_prim_points_2d(self.root_dir, self.obj_class)
                self.max_length = get_max_length(self.labels, self.obj_class)
        else:
            self.depth = load_depth(self.phase, self.root_dir) #sublime, self.obj_class)
            self.labels = load_labels_from_t7(self.phase, self.root_dir, intervals)

        self.init = get_init(self.labels)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, item):
        if opt.init_in is not None:
            prim_init = self.init
        elif opt.reg_init != 'None':
            prim_init = torch.zeros(100)
        else:
            prim_init = copy.deepcopy(self.test_init['test_sample'][:, item])
        if 'SYNTestSet' in opt.check:
            print('SYNTestSet prim_init', prim_init)
        dim = 4 * opt.xyz + 1
        # input_mat = torch.Tensor([prim_init[0], prim_init[1], 0])
        input_mat = torch.cat((torch.Tensor(prim_init[0:2 * opt.xyz]), torch.Tensor([0])), dim=0)
        if opt.init_source == 'given':
            rot_mat = torch.Tensor([prim_init[2], prim_init[3]])
        elif opt.out_r == 'class':
            rot_mat = torch.Tensor(prim_init[2 * opt.xyz + 1:2 * opt.xyz + 3]).long()
            dim = 2 * opt.xyz + 3
        else:
            # rot_mat = torch.Tensor([prim_init[3], prim_init[4]])
            rot_mat = torch.Tensor(prim_init[2 * opt.xyz + 1:4 * opt.xyz + 1])
        cls_mat = torch.zeros(1).long()
        c_mask_mat = torch.ones(1).long()
        if opt.loss_c is not None:
            dim += 1
            cls_mat = torch.Tensor([prim_init[dim - 1]]).long()
        bbox = np.zeros(opt.n_sem * opt.n_para)
        existence = np.zeros(opt.n_sem)
        prim_box_2d = torch.zeros(4)
        box_proposal = torch.zeros(1)
        box_gt_idx = torch.zeros(1)
        if opt.loss_box2d is not None or opt.box2d_source is not None:
            dim += 4
            prim_box_2d = torch.Tensor(prim_init[dim - 4:dim])
        length = torch.zeros(1)
        if opt.encoder in ['resnet', 'hg', 'depth_new']:
            image_name = self.image_names[item]
            id_img_ori = int(image_name[0:4])
            image = load_image(image_name, self.root_dir)
            length[0] = self.length_all[item, 0]
            # bbox_2d = copy.deepcopy(self.bbox_2d[:, id_img_ori - 1])
            # image = self.crop_bbox(image, bbox_2d)
            # sample= transform_image_label(id_img_ori, image, transform=self.transform)
            if opt.bbox_con[:3] == 'ora' or opt.filter:
                voxel_id = get_model_id(id_img_ori, self.match_id)  # 0-215
                part, existence = get_bbox_3d_ins(self.bbox_3d[voxel_id, 0]['sem'][0, 0])
                bbox = normalize_bbox(part, self.mean_std_init)
            if opt.box2d_source is not None:
                prim_box_2d_all, sample = get_prim_box_2d(item, self.match_id, id_img_ori, self.prim_points_2d,
                                                      self.max_length, image, self.transform, inverse=self.inverse)
                if 'SYNTestSet' in opt.check:
                    print('SYNTestSet before', prim_box_2d_all)
                if opt.box2d_source != 'oracle':
                    for j in range(3//opt.xyz):
                        # pass
                        prim_box_2d_all[:, j] = prim_box_2d
                prim_box_2d = prim_box_2d_all
            else:
                sample = transform_image_label(id_img_ori, image, transform=self.transform)
            if opt.encoder == 'resnet':
                if '3dprnnnyu' not in self.obj_class:
                    assert torch.sum(sample['image'] > 1.) == 0 and torch.sum(sample['image'] < 0.) == 0
                sample['image'] = self.normalize(sample['image'])
            if 'SYNTestSet' in opt.check:
                print('SYNTestSet output', input_mat, rot_mat, cls_mat, c_mask_mat, bbox, existence, prim_box_2d)
            return input_mat, rot_mat, cls_mat, c_mask_mat, sample['image'], \
                   bbox, existence, prim_box_2d, length, torch.Tensor([item]), box_proposal, box_gt_idx
        else:
            depth_mat = copy.deepcopy(self.depth[item, :, :])  # (64, 64)
            depth_mat = torch.from_numpy(depth_mat[np.newaxis, :, :])  # (1, 64, 64)
            return input_mat, rot_mat, cls_mat, c_mask_mat, depth_mat, \
                   bbox, existence, prim_box_2d, length, torch.Tensor([item]), box_proposal, box_gt_idx

    def __len__(self):
        if opt.encoder in ['resnet', 'hg', 'depth_new']:
            return len(self.image_names)
        elif opt.encoder == 'depth':
            return self.test_init['test_sample'].shape[1]
        else:
            return 100

    def get_mean_std(self):
        return (self.test_init['mean_x'], self.test_init['mean_y'], self.test_init['mean_r'],
                self.test_init['std_x'], self.test_init['std_y'], self.test_init['std_r'])


class SYNSaveNNSet(Dataset):
    def __init__(self, inverse, transform=None, phase='train'):
        super(SYNSaveNNSet, self).__init__()
        global opt
        opt = get_opt()
        self.inverse = inverse
        self.transform = transform
        self.phase = phase
        self.root_dir = opt.data_dir  # os.path.join(opt.data_dir, opt.obj_class)
        if phase[:4] == 'test':
            self.obj_class = phase[5:]
            if 'all' in opt.obj_class:
                self.root_dir = os.path.join(self.root_dir, self.obj_class)
        else:
            self.obj_class = opt.obj_class
        if opt.encoder in ['resnet', 'hg', 'depth_new']:
            self.image_names = load_names(self.phase, self.root_dir, self.obj_class)
            # self.bbox_2d = self.load_bbox_2d('bboxes.mat')
            self.match_id = load_model_id(self.root_dir, self.obj_class)
            if opt.loss_box2d is not None or opt.box2d_source is not None:
                self.prim_points_2d = load_prim_points_2d(self.root_dir, self.obj_class)
        else:
            self.depth = load_depth(self.phase, self.root_dir) #sublime, self.obj_class)
        # self.labels = load_labels_from_t7(self.phase, self.root_dir, opt.intervals)
        self.labels = load_primset_to_t7_format(self.root_dir, self.obj_class)
        self.max_length = get_max_length(self.labels, self.obj_class)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, item):
        if 'all' in self.obj_class:
            return self.class_agnostic(item)
        prim_box_2d = torch.zeros(4, self.max_length)
        if opt.encoder in ['resnet', 'hg', 'depth_new']:
            image_name = self.image_names[item]
            id_img_ori = int(image_name[0:4])
            image = load_image(image_name, self.root_dir)
            # bbox_2d = copy.deepcopy(self.bbox_2d[:, id_img_ori - 1])
            # image = self.crop_bbox(image, bbox_2d)
            if opt.loss_box2d is not None or opt.box2d_source is not None:
                prim_box_2d, sample = get_prim_box_2d(item, self.match_id, id_img_ori, self.prim_points_2d,
                                                      self.max_length, image, self.transform, inverse=self.inverse)
            else:
                sample = transform_image_label(id_img_ori, image, transform=self.transform)
            if opt.encoder == 'resnet':
                assert torch.sum(sample['image'] > 1.) == 0 and torch.sum(sample['image'] < 0.) == 0
                sample['image'] = self.normalize(sample['image'])
            if 'SYNSaveNNSet' in opt.check:
                print('SYNSaveNNSet output', prim_box_2d)
            return sample['image'], prim_box_2d
        else:
            depth_mat = copy.deepcopy(self.depth[item, :, :])  # (64, 64)
            depth_mat = torch.from_numpy(depth_mat[np.newaxis, :, :])  # (1, 64, 64)
            return depth_mat, prim_box_2d

    def __len__(self):
        if opt.encoder in ['resnet', 'hg', 'depth_new']:
            if 'all' in self.obj_class:
                length = 0
                for cls in opt.file_names['obj_classes']:
                    length += len(self.image_names[cls])
                return length
            return len(self.image_names)
        else:
            return self.depth.shape[0]

    def class_agnostic(self, item):
        item, cls = get_obj_class(item, self.phase)
        prim_box_2d = torch.zeros(4, self.max_length)
        if opt.encoder in ['resnet', 'hg', 'depth_new']:
            image_name = self.image_names[cls][item]
            id_img_ori = int(image_name[0:4])
            image = load_image(image_name, os.path.join(self.root_dir, cls))
            # bbox_2d = copy.deepcopy(self.bbox_2d[:, id_img_ori - 1])
            # image = self.crop_bbox(image, bbox_2d)
            if opt.loss_box2d is not None or opt.box2d_source is not None:
                prim_box_2d, sample = get_prim_box_2d(item, self.match_id[cls], id_img_ori, self.prim_points_2d[cls],
                                                      self.max_length, image, self.transform, inverse=self.inverse)
            else:
                sample = transform_image_label(id_img_ori, image, transform=self.transform)
            if opt.encoder == 'resnet':
                assert torch.sum(sample['image'] > 1.) == 0 and torch.sum(sample['image'] < 0.) == 0
                sample['image'] = self.normalize(sample['image'])
            if 'SYNSaveNNSet' in opt.check:
                print('SYNSaveNNSet output', prim_box_2d)
            return sample['image'], prim_box_2d


class NNCompute(object):
    def __init__(self, inverse):
        global opt
        opt = get_opt()
        self.inverse = inverse
        if opt.save_w_vector:
            self.path_w = opt.init_source
        else:
            self.path_w = '/Users/heqian/Research/1112/projects/3dprnn/intact/torch_test'
        self.root_dir = opt.data_dir
        # self.path_mean_std = opt.data_dir + '/chair' #'/Users/heqian/Research/projects/primitive-based_3d/data/chair'
        self.path_mean_std = self.root_dir   #'/Users/heqian/Research/projects/primitive-based_3d/data/3dprnn'
        # self.phases = ['train', 'val', 'test_chair', 'test_table', 'test_night_stand']
        self.obj_class = opt.obj_class
        self.phases = ['train', 'val'] + ['test_' + x for x in opt.file_names['obj_classes']]
        self.gap = opt.test_gap # 10 or 1
        self.depth_codes, self.lengths, self.depth_codes_train_val, self.labels_train_val, self.box2d_saved = self.load_data()
        # if opt.rgb_con:
        if opt.encoder in ['resnet', 'hg', 'depth_new']:
            self.image_names = {x: load_names(x, self.root_dir, self.obj_class) for x in self.phases}
            #pdb.set_trace()
            self.match_id = load_model_id(self.root_dir, self.obj_class)

    def load_data(self):
        if opt.gt_init:
            return None, None, None, None, None
        nn_saved = {x: scipy.io.loadmat(self.path_w + '/w_vector_{}.mat'.format(x)) for x in self.phases}
        box2d_saved = {x: nn_saved[x]['box2d'] for x in self.phases}
        depth_codes = {x: nn_saved[x]['x'] for x in self.phases}
        lengths = {x: depth_codes[x].shape[0] for x in self.phases}
        depth_codes_train_val = np.concatenate((depth_codes['train'], depth_codes['val']), axis=0)
        # labels_train_val = {x: load_labels_from_t7(x, self.root_dir) for x in ['train', 'val']}
        labels_train_val = {x: load_primset_to_t7_format(self.root_dir, self.obj_class) for x in ['train', 'val']}
        return depth_codes, lengths, depth_codes_train_val, labels_train_val, box2d_saved

    def init_from_gt_all_class(self):
        # phases = ['test_chair', 'test_table', 'test_night_stand']
        phases = ['test_' + x for x in opt.file_names['obj_classes']]
        for phase in phases:
            gt_dir = os.path.join(self.root_dir, 'prim_gt', 'prim_sort_mn_{}_test.mat'.format(phase[5:]))
            gt = scipy.io.loadmat(gt_dir)['primset']    # (100, 1)
            self.init_from_gt_one_class(phase, gt)

    def init_from_gt_one_class(self, phase, gt):
        length = gt.shape[0] * 50   # 5000, 5000, 4300
        nn_init = torch.zeros(length, 5)
        nn_retrieve = torch.zeros(length, 5, opt.max_len)
        nn_id = -torch.ones(length, 1)
        # file_name = 'sample_generation/test_NNfeat_mn_{}.mat'.format(phase[5:])
        # file_name = opt.file_names['mean_std'][phase[5:]]
        # mean_std_init = scipy.io.loadmat(os.path.join(self.path_mean_std, file_name))
        mean_std_init = opt.mean_std
        for i in range(0, length, self.gap):
            nn_id[i, 0] = i
            label = gt[i // 50, 0][0, 0]['ori'][0, 10:]
            label_normalized = self.normalize_gt(label, mean_std_init)
            nn_init[i, :] = copy.deepcopy(label_normalized)
        mean_std_init['test_sample'] = nn_init.transpose(0, 1).numpy()
        mean_std_init['test_ret_num'] = nn_id.numpy()   # np.zeros((depth_codes_test.shape[0], 1))
        mean_std_init['nn_res'] = nn_retrieve.numpy()
        scipy.io.savemat(opt.init_source + '/test_NNfeat_mn_{}_l2.mat'.format(phase[5:]), mean_std_init)

    def normalize_gt(self, label, mean_std_init):
        label_normalized = torch.ones(5)
        label_normalized[0] = label[0]
        label_normalized[1] = label[3]
        label_normalized[2] = 0
        label_normalized[3] = label[9] * label[6]
        label_normalized[4] = label[6]
        # rs = torch.zeros(4, opt.max_len)
        mean_x = mean_std_init['mean_x']
        mean_y = mean_std_init['mean_y']
        mean_r = mean_std_init['mean_r']
        std_x = mean_std_init['std_x']
        std_y = mean_std_init['std_y']
        std_r = mean_std_init['std_r']
        if not opt.global_denorm:
            label_normalized[0] = (label_normalized[0] - mean_x[0, 0]) / std_x[0, 0]
            label_normalized[1] = (label_normalized[1] - mean_y[0, 0]) / std_y[0, 0]
            label_normalized[3] = (label_normalized[3] - mean_r[0, 0]) / std_r[0, 0]

        # x[:, 0] = x[:, 0] * std_x[0, 0] + mean_x[0, 0]
        # x[:, 1] = x[:, 1] * std_y[0, 0] + mean_y[0, 0]
        # r[:, 0] = r[:, 0] * std_r[0, 0] + mean_r[0, 0]
        return label_normalized

    def compute_nn_all_class(self):
        # phases = ['test_chair', 'test_table', 'test_night_stand']
        phases = ['test_' + x for x in opt.file_names['obj_classes']]
        for phase in phases:
            self.compute_nn_one_class(phase)

    def compute_nn_one_class(self, phase):
        # file_name = opt.file_names['mean_std'][phase[5:]]
        # mean_std_init = scipy.io.loadmat(os.path.join(self.path_mean_std, file_name))
        mean_std_init = opt.mean_std
        n_dim = 2*opt.xyz + 1
        if opt.out_r == 'class':
            n_dim += 2
        else:
            n_dim += 2*opt.xyz
        if opt.loss_c is not None:
            n_dim += 1
        if opt.loss_box2d is not None or opt.box2d_source is not None:
            n_dim += 4
        nn_init = torch.zeros(self.lengths[phase], n_dim)
        nn_retrieve = torch.zeros(self.lengths[phase], n_dim, opt.max_len)
        nn_id = -torch.ones(self.lengths[phase], 1)
        # if self.lengths[phase] > 500:
        if 'ts' in opt.test_version or opt.test_version == '1':
            code_length = self.lengths['train']+self.lengths['val']
            print(self.lengths[phase], 'train+val')
        else:
            code_length = self.lengths['train']
            print(self.lengths[phase], 'train only')
        for i in range(0, self.lengths[phase], self.gap):
            # m_dis_cos = -1e10
            # m_id_cos = -1
            m_dis_l2 = 1e10
            m_id_l2 = -1
            v1 = copy.deepcopy(self.depth_codes[phase][i, :])
            # model_id_i = self.get_m_id(i)
            for j in range(code_length):
                # model_id_j = self.get_m_id(j)
                # if (phase == 'train' or phase == 'val') and model_id_i == model_id_j:
                #     continue
                v2 = copy.deepcopy(self.depth_codes_train_val[j, :])
                # dis_cos = self.__cos_distance(v1, v2)
                dis_l2 = self.__l2_distance(v1, v2)
                # if dis_cos > m_dis_cos: # > cosine similarity, < l2 distance
                #     m_dis_cos = dis_cos
                #     m_id_cos = j
                if dis_l2 < m_dis_l2:
                    m_dis_l2 = dis_l2
                    m_id_l2 = j
            m_id = m_id_l2
            if opt.encoder in ['resnet', 'hg', 'depth_new']:
                if 'all' in opt.obj_class:
                    m_id, model_id, train_or_val, cls = self.get_m_id_all_class(m_id)
                    nn_id[i, 0] = float(model_id)
                    label = self.labels_train_val['train'][cls][model_id]
                    box2d = self.box2d_saved[train_or_val][m_id]
                else:
                    m_id, model_id, train_or_val = self.get_m_id(m_id)
                    nn_id[i, 0] = float(model_id)
                    label = self.labels_train_val['train'][model_id]
                    box2d = self.box2d_saved[train_or_val][m_id]
            else:
                nn_id[i, 0] = m_id  # 0-7404
                if m_id < self.lengths['train']:
                    label = self.labels_train_val['train'][m_id]
                else:
                    m_id = m_id - self.lengths['train']
                    label = self.labels_train_val['val'][m_id]
                box2d = np.zeros(4)
            train_val_res = self.get_item(label, n_dim, mean_std_init)
            box2d = torch.from_numpy(box2d)
            if opt.loss_box2d is not None or opt.box2d_source is not None:
                train_val_res[-4:, 0] = box2d
            nn_init[i, :] = copy.deepcopy(train_val_res[:, 0])
            nn_retrieve[i, :, :] = copy.deepcopy(train_val_res)
        if phase == 'train' or phase == 'val':
            mean_std_init = {}
            mean_std_init['test_sample'] = nn_init.transpose(0, 1).numpy()
            mean_std_init['test_ret_num'] = nn_id.numpy()  # np.zeros((depth_codes_test.shape[0], 1))
            mean_std_init['nn_res'] = nn_retrieve.numpy()
            scipy.io.savemat(opt.exp_dir + '/{}_nn_init.mat'.format(phase), mean_std_init)
        else:
            if 'all' in opt.obj_class:
                mean_std_init = mean_std_init[phase[5:]]
            mean_std_init['test_sample'] = nn_init.transpose(0, 1).numpy()
            mean_std_init['test_ret_num'] = nn_id.numpy()   # np.zeros((depth_codes_test.shape[0], 1))
            mean_std_init['nn_res'] = nn_retrieve.numpy()
            scipy.io.savemat(opt.init_source + '/test_NNfeat_mn_{}_l2.mat'.format(phase[5:]), mean_std_init)

    def get_m_id(self, m_id):
        if m_id < self.lengths['train']:
            image_name = self.image_names['train'][m_id]
            train_or_val = 'train'
        else:
            m_id = m_id - self.lengths['train']
            image_name = self.image_names['val'][m_id]
            train_or_val = 'val'
        id_img_ori = int(image_name[0:4])
        model_id = get_model_id(id_img_ori, self.match_id)
        return m_id, model_id, train_or_val

    def get_m_id_all_class(self, m_id):
        if m_id < self.lengths['train']:
            m_id_cls, cls = get_obj_class(m_id, 'train')
            image_name = self.image_names['train'][cls][m_id_cls]
            train_or_val = 'train'
        else:
            m_id = m_id - self.lengths['train']
            m_id_cls, cls = get_obj_class(m_id, 'val')
            image_name = self.image_names['val'][cls][m_id_cls]
            train_or_val = 'val'
        id_img_ori = int(image_name[0:4])
        model_id = get_model_id(id_img_ori, self.match_id[cls])
        return m_id, model_id, train_or_val, cls

    def get_item(self, label, n_dim, mean_std_init):
        ins_length = len(label[b'x_vals']) // opt.xyz
        input_rot_mat = torch.zeros(n_dim, opt.max_len)
        # input_rot_mat[0, :ins_length] = torch.Tensor(label[b'x_vals'])
        # input_rot_mat[1, :ins_length] = torch.Tensor(label[b'y_vals'])
        # input_rot_mat[2, :ins_length] = torch.Tensor(label[b'e_vals'])
        # input_rot_mat[3, :ins_length] = torch.Tensor(label[b'r_vals'])
        # input_rot_mat[4, :ins_length] = torch.Tensor(label[b'rs_vals'])
        input_mat, rot_mat, cls_mat, _, _ = get_label(-1, label, opt.max_len//3*3, test_init=mean_std_init,
                                                      inverse=self.inverse)
        input_rot_mat[:2*opt.xyz+1, :ins_length] = input_mat[:, :ins_length]
        if opt.out_r == 'class':
            input_rot_mat[2 * opt.xyz + 1:2 * opt.xyz + 3, :ins_length] = rot_mat[:, :ins_length]
            cls_start = 2 * opt.xyz + 3
        else:
            input_rot_mat[2 * opt.xyz + 1:4 * opt.xyz + 1, :ins_length] = rot_mat[:, :ins_length]
            cls_start = 4 * opt.xyz + 1
        if opt.loss_c is not None:
            # input_rot_mat[5, :ins_length] = torch.Tensor(label[b'cls']) - 1
            input_rot_mat[cls_start, :ins_length] = cls_mat[:, :ins_length]
        return input_rot_mat

    def __cos_distance(self, v1, v2):
        dis = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        if dis > 10:
            dis = -1
        return dis

    def __l2_distance(self, v1, v2):
        dis = np.sqrt(np.sum((v1 - v2) * (v1 - v2)))
        return dis

    def compute_nn_all_class_0(self):
        # classes = ['chair', 'table', 'night_stand']
        classes = opt.file_names['obj_classes']
        for obj_class in classes:
            self.compute_nn_torch_one_class(obj_class)

    def compute_nn_torch_one_class(self, obj_class=None):
        mean_std_init = opt.mean_std
        n_dim = 5
        if opt.loss_c is not None:
            n_dim = 6
        # if obj_class is None:
        #     obj_class = opt.obj_class
        # dirs = 'init'
        # path_w = '/Users/heqian/Research/1112/projects/3dprnn/intact/torch_test'
        depth_codes_train = scipy.io.loadmat(self.path_w + '/w_vector_train.mat')['x']
        depth_codes_val = scipy.io.loadmat(self.path_w + '/w_vector_val.mat')['x']
        depth_codes_test = scipy.io.loadmat(self.path_w + '/w_vector_test_{}.mat'.format(obj_class))['x']
        depth_codes_train_val = np.concatenate((depth_codes_train, depth_codes_val), axis=0)
        # labels_train_val = {x: load_labels_from_t7(x, self.root_dir) for x in ['train', 'val']}
        labels_train_val = {x: load_primset_to_t7_format(self.root_dir, self.obj_class) for x in ['train', 'val']}
        nn_init = torch.zeros(depth_codes_test.shape[0], n_dim)
        nn_retrieve = torch.zeros(depth_codes_test.shape[0], n_dim, opt.max_len)
        nn_id = torch.zeros(depth_codes_test.shape[0], 1)
        for i in range(0, depth_codes_test.shape[0], self.gap):
            # m_dis_cos = -1e10
            # m_id_cos = -1
            m_dis_l2 = 1e10
            m_id_l2 = -1
            v1 = copy.deepcopy(depth_codes_test[i, :])
            for j in range(depth_codes_train_val.shape[0]):
                v2 = copy.deepcopy(depth_codes_train_val[j, :])
                # dis_cos = self.__cos_distance(v1, v2)
                dis_l2 = self.__l2_distance(v1, v2)
                # if dis_cos > m_dis_cos: # > cosine similarity, < l2 distance
                #     m_dis_cos = dis_cos
                #     m_id_cos = j
                if dis_l2 < m_dis_l2:
                    m_dis_l2 = dis_l2
                    m_id_l2 = j
            # print(m_dis_cos, m_id_cos, m_dis_l2, m_id_l2)
            # print('='*10, i, m_dis_l2, m_id_l2)
            m_id = m_id_l2
            nn_id[i, 0] = m_id
            if m_id < depth_codes_train.shape[0]:
                label = labels_train_val['train'][m_id]
            else:
                m_id = m_id - depth_codes_train.shape[0]
                label = labels_train_val['val'][m_id]
            train_val_res = self.get_item(label, n_dim, mean_std_init)
            nn_init[i, :] = copy.deepcopy(train_val_res[:, 0])
            nn_retrieve[i, :, :] = copy.deepcopy(train_val_res)
        # path = '/Users/heqian/Research/projects/primitive-based_3d/data/3dprnn'
        # file_name = 'sample_generation/test_NNfeat_mn_{}.mat'.format(obj_class)
        # file_name = opt.file_names['mean_std'][obj_class]
        # mean_std_init = scipy.io.loadmat(os.path.join(self.path_mean_std, file_name))
        mean_std_init = opt.mean_std
        mean_std_init['test_sample'] = nn_init.transpose(0, 1).numpy()
        mean_std_init['test_ret_num'] = nn_id.numpy()   # np.zeros((depth_codes_test.shape[0], 1))
        mean_std_init['nn_res'] = nn_retrieve.numpy()
        scipy.io.savemat(opt.init_source + '/test_NNfeat_mn_{}_l2.mat'.format(obj_class), mean_std_init)


class SaveGTt7(object):
    def __init__(self):
        global opt
        opt = get_opt()
        self.root_dir = opt.data_dir

    def save_train_val_gt_from_t7(self):
        t7_dir = os.path.join(self.root_dir, 'gt_pth')
        if not os.path.exists(t7_dir):
            os.mkdir(t7_dir)
        # test_classes = ['test_chair', 'test_table', 'test_night_stand']
        test_classes = ['test_' + x for x in opt.file_names['obj_classes']]
        datasets = {x: SYNTestSet(inverse=opt.inverse, phase=x) for x in test_classes}
        for phase in ['train', 'val']:
            labels = load_labels_from_t7(phase, self.root_dir)
            rs_res = torch.zeros(4 * len(labels), opt.max_len)
            cls_res = - torch.ones(len(labels), opt.max_len).long()
            for i in range(len(labels)):
                label = labels[i]
                ins_length, input_mat, rot_mat, cls_mat = self.get_label_ins(label)
                test_class = self.get_test_class(phase, i)
                outputs = []
                for j in range(ins_length):
                    outputs.append((input_mat[:, :, j], rot_mat[:, :, j], cls_mat[:, :, j]))
                rs, cls = self.outputs_to_results_t7(outputs, datasets[test_class])
                rs_res[i * 4: i * 4 + 4, :] = rs
                cls_res[i, :] = cls
                print('saving gt from .t7', phase, i)
            scipy.io.savemat(t7_dir + '/test_res_mn_{}.mat'.format(phase),
                             {'x': rs_res.numpy(), 'cls': cls_res.numpy()})

    def outputs_to_results_t7(self, outputs, dataset):
        rs = torch.zeros(4, opt.max_len)
        cls = - torch.ones(1, opt.max_len).long()
        mean_x, mean_y, mean_r, std_x, std_y, std_r = dataset.get_mean_std()
        for i in range(len(outputs)):
            x, r, c = outputs[i]
            if not opt.global_denorm:
                x[:, 0] = x[:, 0] * std_x[0, 0] + mean_x[0, 0]
                x[:, 1] = x[:, 1] * std_y[0, 0] + mean_y[0, 0]
                r[:, 0] = r[:, 0] * std_r[0, 0] + mean_r[0, 0]

            assert x.size()[0] == 1 and r.size()[0] == 1# and c.size()[0] == 1
            rs[0, i] = x[0, 0]
            rs[1, i] = x[0, 1]
            rs[2, i] = r[0, 0]
            rs[3, i] = r[0, 1]
            # print(rs[:, :i+2])
            # if opt.loss_c is not None:
            #     if i % 3 == 0:
            #         cls[0, i] = c[0, 0] # class, not prob
            if opt.loss_c is not None or opt.len_adjust:
                if opt.init_in is None: # not examined
                    cls_i = i
                    cls_ii = i
                else:
                    cls_i = i + 1
                    cls_ii = i - 2
                if cls_i % 3 == 0:
                    cls[0, cls_ii] = c[0, 0] # class, not prob
            if x[0, 2] == 1:
                break
        return rs, cls

    def get_test_class(self, phase, i):
        # test_classes = ['test_chair', 'test_table', 'test_night_stand']
        test_classes = ['test_' + x for x in opt.file_names['obj_classes']]
        intervals = opt.intervals
        if i < intervals[phase][0]:
            test_class = test_classes[0]
        elif i < intervals[phase][1]:
            test_class = test_classes[1]
        elif i < intervals[phase][2]:
            test_class = test_classes[2]
        else:
            raise NotImplementedError
        return test_class

    def get_label_ins(self, label):
        ins_length = len(label[b'x_vals'])
        input_mat = torch.zeros(3, ins_length)
        rot_mat = torch.zeros(2, ins_length)
        cls_mat = torch.zeros(1, ins_length)
        input_mat[0, :] = torch.Tensor(label[b'x_vals'])
        input_mat[1, :] = torch.Tensor(label[b'y_vals'])
        input_mat[2, :] = torch.Tensor(label[b'e_vals'])
        rot_mat[0, :] = torch.Tensor(label[b'r_vals'])
        rot_mat[1, :] = torch.Tensor(label[b'rs_vals'])
        if opt.loss_c is not None:
            cls_mat[0, :] = torch.Tensor(label[b'cls']) - 1
        input_mat = torch.unsqueeze(input_mat, dim=0)
        rot_mat = torch.unsqueeze(rot_mat, dim=0)
        cls_mat = torch.unsqueeze(cls_mat, dim=0)
        return ins_length, input_mat, rot_mat, cls_mat.long()


class BBox3DDataset(Dataset):
    def __init__(self, trans=None, phase='train'):
        global opt
        opt = get_opt()
        self.transform = trans
        self.phase = phase
        self.root_dir = opt.data_dir
        self.obj_class = opt.obj_class
        self.image_names = load_names(self.phase, self.root_dir, self.obj_class)
        # self.bbox_2d = self.load_bbox_2d('bboxes.mat')
        # self.bbox_3d = load_bbox_3d('Myprimset_paired.mat', self.root_dir)
        self.bbox_3d = load_bbox_3d(opt.file_names['primset'], self.root_dir)
        self.match_id = load_model_id(self.root_dir, self.obj_class)
        # self.label_info = json.load(open(os.path.join(self.root_dir, 'pix3d.json', 'r')))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        image_name = self.image_names[item]
        id_img_ori = int(image_name[0:4])
        image = load_image(image_name, self.root_dir)
        voxel_id = get_model_id(id_img_ori, self.match_id)  # 0-215
        part, existence = get_bbox_3d_ins(self.bbox_3d[voxel_id, 0]['sem'][0, 0])
        # part, existence = get_bbox_3d_ins(self.bbox_3d[id_img_ori - 1, 0]['sem'][0, 0])
        # part, existence = self.get_label(id_img_ori)
        # bbox_2d = copy.deepcopy(self.bbox_2d[:, id_img_ori - 1])
        # image = self.crop_bbox(image, bbox_2d)
        sample = transform_image_label(id_img_ori, image, label=part, transform=self.transform)
        return sample, existence
