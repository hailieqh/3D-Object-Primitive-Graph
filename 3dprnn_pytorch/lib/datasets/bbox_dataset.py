import cv2
import os
import copy
import math
import random
import numpy as np
import scipy.io
import torch
import torchvision.transforms
from torch.utils.data import Dataset
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from skimage import io, transform

# from torchvision.transforms.transforms import functional as F
from lib.opts import *
# from lib.utils.utils import *


class PadSquare(object):
    def __call__(self, sample):
        global opt
        opt = get_opt()
        image, label = sample['image'], sample['label']
        # print('before pad', type(image[0,0,0]), np.max(image))
        label = copy.deepcopy(label)
        h, w, c = image.shape
        tmp = image
        if h < w:
            pad = (w - h) // 2
            tmp = np.zeros((w, w, c))
            tmp[pad : pad+h, :, :] = copy.deepcopy(image)
        elif h > w:
            pad = (h - w) // 2
            tmp = np.zeros((h, h, c))
            tmp[:, pad : pad+w, :] = copy.deepcopy(image)
        else:
            pad = 0
        if opt.obj_class[:6] != '3dprnn':
            image = tmp.astype(np.uint8)
        else:
            image = tmp
        # print('after pad', type(image[0, 0, 0]), np.max(image))
        if (opt.canonical and opt.loss_box2d is None and opt.box2d_source is None) or label is None:
            return {'image': image, 'label': label}

        if h < w:
            if image.shape == label.shape:  # depth
                tmp_label = np.zeros((w, w, c))
                tmp_label[pad : pad+h, :, :] = copy.deepcopy(label)
                label = tmp_label
            else:   # kp
                label[:, 1] += pad
        elif h > w:
            if image.shape == label.shape:  # depth
                tmp_label = np.zeros((h, h, c))
                tmp_label[:, pad : pad+w, :] = copy.deepcopy(label)
                label = tmp_label
            else:   # kp
                label[:, 0] += pad
        return {'image': image, 'label': label}


class RandomRotate(object):
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        label = copy.deepcopy(label)
        size = img.shape[0]
        a = random.random()
        b = random.random()
        if a < 0.5:
            rot = 15 * b
        else:
            rot = -15 * b
        img = Image.fromarray(np.uint8(img))
        img = img.rotate(rot)
        img = np.asarray(img)
        if opt.canonical:
            return {'image': img, 'label': label}

        if img.shape == label.shape:
            tmp_label = copy.deepcopy(label[:, :, 0])
            tmp_label = Image.fromarray(tmp_label)
            tmp_label = tmp_label.rotate(rot)
            tmp_label = np.asarray(tmp_label)
            tmp_label = np.array([tmp_label, tmp_label, tmp_label]).transpose((1, 2, 0))
            outputs = tmp_label
        else:
            center = np.ones(2) * size / 2
            pts = label
            num = pts.shape[0]
            outputs = np.zeros(pts.shape)
            r = np.eye(3)
            ang = -rot * math.pi / 180
            s = math.sin(ang)
            c = math.cos(ang)
            r[0][0] = c
            r[0][1] = -s
            r[1][0] = s
            r[1][1] = c
            for i in range(num):
                pt = np.ones(3)
                pt[0], pt[1] = pts[i][0]-center[0], pts[i][1]-center[1]
                new_point = np.dot(r, pt)
                outputs[i] = new_point[0:2] + center
        
        return {'image': img, 'label': outputs}


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        img, label = sample['image'], sample['label']
        label = copy.deepcopy(label)
        h, w, c = img.shape
        #if True:
        if random.random() < 0.5:
            img = Image.fromarray(np.uint8(img))
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = np.asarray(img)
            sample['image'] = img
            if opt.canonical:
                return {'image': sample['image'], 'label': label}

            if img.shape == label.shape:
                tmp_label = copy.deepcopy(label[:, :, 0])
                tmp_label = Image.fromarray(tmp_label)
                tmp_label = tmp_label.transpose(Image.FLIP_LEFT_RIGHT)
                tmp_label = np.asarray(tmp_label)
                tmp_label = np.array([tmp_label, tmp_label, tmp_label]).transpose((1, 2, 0))
                label = tmp_label
            else:
                label[:, 0] = w - label[:, 0]
                n = label.shape[0] #opt.nKeypoints
                nn = int(n / 2)
                if opt.dataset == 'bed' or opt.dataset == 'sofa' or opt.dataset == 'car':
                    tmp = copy.deepcopy(label[0:nn, :])
                    label[0:nn, :] = copy.deepcopy(label[nn:n, :])
                    label[nn:n, :] = copy.deepcopy(tmp)
                if opt.dataset == 'chair' or opt.dataset == 'table':
                    for i in range(nn):
                        j = i * 2
                        tmp = copy.deepcopy(label[j, :])
                        label[j, :] = copy.deepcopy(label[j+1, :])
                        label[j+1, :] = copy.deepcopy(tmp)
                if opt.dataset == 'swivelchair':
                    inter = [4, 2]
                    for i in range(2):
                        tmp = copy.deepcopy(label[i, :])
                        label[i, :] = copy.deepcopy(label[i+inter[i], :])
                        label[i+inter[i], :] = copy.deepcopy(tmp)
                    for i in range(3):
                        j = i * 2 + 7
                        tmp = copy.deepcopy(label[j, :])
                        label[j, :] = copy.deepcopy(label[j+1, :])
                        label[j+1, :] = copy.deepcopy(tmp)
                if opt.dataset == 'flic':
                    for i in range(3):
                        tmp = copy.deepcopy(label[i, :])
                        label[i, :] = copy.deepcopy(label[i+3, :])
                        label[i+3, :] = copy.deepcopy(tmp)
                    for i in range(2):
                        j = i * 2 + 6
                        tmp = copy.deepcopy(label[j, :])
                        label[j, :] = copy.deepcopy(label[j+1, :])
                        label[j+1, :] = copy.deepcopy(tmp)
            sample['label'] = label

        return sample


class RandomCut(object):
    def __init__(self, edge):
        self.edge = edge

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        label = copy.deepcopy(label)
        canonical_label = copy.deepcopy(label)
        h, w, c = image.shape
        n, d = label.shape
        edge = int(self.edge * random.random())
        if w >= h:
            left = label[:, 0] - 2*edge
            right = w - (label[:, 0] + 2*edge)
            if random.random() < 0.5 and np.sum(np.abs(left)) == np.sum(left):
                image = image[:, edge:, :]
                label[:, 0] = label[:, 0] - edge
            if random.random() < 0.5 and np.sum(np.abs(right)) == np.sum(right):
                image = image[:, :w-edge, :]
                label[:, 0] = label[:, 0] + edge
        else:
            up = label[:, 1] - 2*edge
            down = h - (label[:, 1] + 2*edge)
            if random.random() < 0.5 and np.sum(np.abs(up)) == np.sum(up):
                image = image[edge:, :, :]
                label[:, 1] = label[:, 1] - edge
            if random.random() < 0.5 and np.sum(np.abs(down)) == np.sum(down):
                image = image[:h-edge, :, :]
                label[:, 1] = label[:, 1] + edge
        sample['image'], sample['label'] = image, label

        if opt.canonical:
            return {'image': sample['image'], 'label': canonical_label}
        return sample


class RandomRColor(object):
    def __call__(self, sample):
        global opt
        opt = get_opt()
        img, _ = sample['image'], sample['label']
        # print('before color', type(img[0, 0, 0]), np.max(img))
        tmp = np.zeros(img.shape)
        tmp[:, :, 0] = copy.deepcopy(img[:, :, 0])*np.random.uniform(0.6,1.4)
        tmp[:, :, 1] = copy.deepcopy(img[:, :, 1])*np.random.uniform(0.6,1.4)
        tmp[:, :, 2] = copy.deepcopy(img[:, :, 2])*np.random.uniform(0.6,1.4)
        tmp = np.maximum(tmp, 0)
        tmp = np.minimum(tmp, 255)
        tmp = tmp.astype(np.uint8)
        sample['image'] = tmp
        # print('after color', type(img[0, 0, 0]), np.max(img))
        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # print('before rescale', type(image[0, 0, 0]), np.max(image))
        label = copy.deepcopy(label)
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        # print('before', np.sum(image), image.shape)
        img = transform.resize(image, (new_h, new_w), mode='constant', anti_aliasing=False)
        # print('after', np.sum(image), image.shape)
        # interpolation=Image.BILINEAR
        # img = F.resize(image, self.output_size, interpolation)
        # print('after rescale', type(img[0, 0, 0]), np.max(img))
        if (opt.canonical and opt.loss_box2d is None and opt.box2d_source is None) or label is None:
            return {'image': img, 'label': label}

        if image.shape == label.shape:
            label = transform.resize(label, (new_h / 4.0, new_w / 4.0), mode='constant', anti_aliasing=False)
            # label = copy.deepcopy(lab)##ugly
        elif opt.loss_box2d is not None or opt.box2d_source is not None:
            label[:, 0] *= new_w / w
            label[:, 1] *= new_h / h
        else:
            # h and w are swapped for label because for images,
            # x and y axes are axis 1 and 0 respectively
            #label = label * [new_w / w, new_h / h]
            label[:, 0] *= 64.0/w
            label[:, 1] *= 64.0/h
        return {'image': img, 'label': label}


class RandomCrop(object): # keypoint (not used)
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        label = copy.deepcopy(label)
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w,
                      :]
        if opt.canonical:
            return {'image': image, 'label': label}
        
        label = label - [left, top]

        return {'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # print('before tensor', type(image[0, 0, 0]), np.max(image))
        label = copy.deepcopy(label) # avoid changing original label

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        if label is None:
            return {'image': torch.from_numpy(image),
                'label': None}
        if image.shape[1] == label.shape[1]*4:  # rescaled depth label, 64
            label = label.transpose((2, 0, 1))
            label = label[0, :, :]
            label = label[np.newaxis, :, :]
        # print('after tensor', type(image[0, 0, 0]), np.max(image))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}
