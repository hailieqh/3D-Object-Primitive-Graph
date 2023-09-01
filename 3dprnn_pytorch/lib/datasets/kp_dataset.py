import os
import math
import random
import copy
import scipy.io
import h5py
from skimage import io, transform
from PIL import Image, ImageOps
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from lib.opts import *
from lib.utils.utils import *
from .prnn_dataset import *


class Pix3DSemKP(Dataset):
    def __init__(self, transform=None, phase='train'):
        global opt
        opt = get_opt()
        self.transform = transform
        self.phase = phase
        self.root_dir = opt.data_dir
        self.obj_class = opt.obj_class
        self.image_names = load_names(self.phase, self.root_dir, self.obj_class)
        self.labels = self.load_labels()
        self.num_sem = self.get_num_sem()
        self.match_id = load_model_id(self.root_dir, self.obj_class)
        # self.gt = self.load_gt(self.root_dir, gt_file)  # (2, 30, 4, 1480)
        # self.bboxes = self.load_bbox(self.root_dir, 'bboxes.mat')
        self.sigma = opt.hmGauss
        self.gauss_kernel = gaussian_kernel_2d(6*self.sigma+1, self.sigma)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        # idx: 0, ..., self.num - 1, idx != or_idx -1
        image_name = self.image_names[item]
        id_img_ori = int(image_name[0:4])
        image = load_image(image_name, self.root_dir)   # 1-3839
        img_id_real = list(self.match_id['img_idxs'][0]).index(id_img_ori)  # 0-3493
        label_kp = np.array(self.labels['after'][str(img_id_real)]['keypoint'])
        label_sem = np.array(self.labels['after'][str(img_id_real)]['sem'])
        if int(self.labels['after'][str(img_id_real)]['name'][:4]) != id_img_ori:
            print(item, self.labels['after'][str(img_id_real)]['name'], str(id_img_ori) + '.png')
        # gt = copy.deepcopy(self.gt[:, :, :, id_img_ori - 1])
        # bbox = copy.deepcopy(self.bboxes[:, id_img_ori - 1])
        # image, gt = self.crop_bbox(image, gt, bbox)
        sample = transform_image_label(id_img_ori, image, label_kp, transform=self.transform)
        label_kp = copy.deepcopy(sample['label'])
        label_kp_same_size = - torch.ones(30, 2)
        label_kp_same_size[:label_kp.size()[0], :] = label_kp
        sample['label'] = self.kp_to_heatmap(sample['label'].numpy(), label_sem)
        return sample, label_kp_same_size

    def load_labels(self):
        file_path = os.path.join(self.root_dir, 'kp2_3_29.json')
        labels = json.load(open(file_path, 'r'))
        return labels

    def get_num_sem(self):
        sem_list = []
        for i in range(len(self.labels['after'])):
            sem_list += self.labels['after'][str(i)]['sem']
        num_sem = np.max(np.array(sem_list))
        return num_sem.astype(np.int)

    def kp_to_heatmap(self, label_kp, label_sem):
        sem_kp = {x: [] for x in range(1, self.num_sem + 1, 1)}
        # import pdb; pdb.set_trace()
        for i in range(label_sem.shape[0]):
            label_sem_i = set(label_sem[i])
            for sem_j in label_sem_i:
                if sem_j == 0:
                    continue
                sem_kp[int(sem_j)].append(label_kp[i, :])
        heatmaps = torch.zeros(self.num_sem, 64, 64)
        for x in sem_kp.keys():
            kp = np.array(sem_kp[x])
            heatmap = draw_gaussian(kp, self.sigma, self.gauss_kernel, kp.shape[0])
            heatmap = heatmap.transpose((2, 0, 1))
            heatmap = torch.from_numpy(heatmap)
            heatmaps[x - 1, :, :] = heatmap
        return heatmaps


def gaussian_kernel_2d(size, sigma):
    sigma_2 = 2 * sigma * sigma
    r_size = (size - 1)//2
    gk = np.zeros((size, size))
    for i in range(-r_size, r_size + 1):
        h = i + r_size
        for j in range(-r_size, r_size + 1):
            w = j + r_size
            v = np.exp(-(i*i + j*j) / sigma_2)
            gk[h, w] = v
            if i*i + j*j > r_size*r_size:
                gk[h, w] = 0
    return gk


def draw_gaussian(label, sigma, gauss_kernel, num_kp):
    # num_kp = opt.nKeypoints # 10
    h, w = opt.output_res, opt.output_res # 64, 64
    out = np.zeros((h, w, 1)) # (64, 64, 1)
    for i in range(num_kp):
        x, y = label[i]
        if x < 0 or x >= 64 or y < 0 or y >= 64:
            continue
        x, y = int(x), int(y)
        r_size = 3 * sigma
        tmp = np.zeros((64 + 2 * r_size, 64 + 2 * r_size))
        tmp[y : y+2*r_size+1, x : x+2*r_size+1] = gauss_kernel
        out[:, :, 0] += tmp[r_size : 64+r_size, r_size : 64+r_size]
    return out
