import numpy as np
import cv2
import os
import scipy.io
import copy
import json
import pdb


def crop_bbox(image, bbox):
    thresh = 0.1
    h, w, _ = image.shape
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min - (x_max - x_min) * thresh)
    x_max = min(w, x_max + (x_max - x_min) * thresh)
    y_min = max(0, y_min - (y_max - y_min) * thresh)
    y_max = min(h, y_max + (y_max - y_min) * thresh)
    # image = copy.deepcopy(image[int(y_min):int(y_max), int(x_min):int(x_max), :])
    if len(image.shape) == 2:
        image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    elif len(image.shape) == 3:
        image = image[int(y_min):int(y_max), int(x_min):int(x_max), :]
    else:
        print('ERROR!!!!!', image.shape)
    return image


def crop_and_resize(in_path, out_path, box2d_all):
    for tmp_path, dirs, files in os.walk(in_path):
        if len(dirs) == 0:
            files = sorted(files)
            for file in files:
                # pdb.set_trace()
                # idx = int(file[0:4])
                # print(idx)
                print(file)
                file_dir = os.path.join(in_path, file)
                image = cv2.imread(file_dir)
                # bboxes_ins_2d_dir = os.path.join(root, 'bboxes.mat')
                # bboxes_ins_2d = scipy.io.loadmat(bboxes_ins_2d_dir)
                # bboxes_ins_2d = bboxes_ins_2d['bboxes']
                # bbox_ins_2d = bboxes_ins_2d[:, idx-1]
                bbox_ins_2d = box2d_all[file]
                image_crop = crop_bbox(image, bbox_ins_2d)
                image_crop_dir = os.path.join(out_path, file[0:4]+'.png')

                h, w, _ = image_crop.shape
                # pdb.set_trace()
                short_edge = min(h, w)
                resize_dim = 320.
                img_cv2_re = cv2.resize(image_crop, (int(w * resize_dim / short_edge), int(h * resize_dim / short_edge)))
                # img_cv2_re = image_crop

                cv2.imwrite(image_crop_dir, img_cv2_re)


def get_box2d(pix3d, cls):
    box2d_all = {}
    for i in range(len(pix3d)):
        if pix3d[i]['category'] == cls:
            name = pix3d[i]['img'].split('/')[-1]
            box2d_all[name] = pix3d[i]['bbox']
    return box2d_all


def crop_and_resize_all():
    # root = '/Users/heqian/Research/projects/primitive-based_3d/data//chair'
    root = os.path.abspath('.')
    cls_all = ['chair', 'bed', 'bookcase', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
    # cls = 'chair'
    pix3d_dir = os.path.join(root, '../input/pix3d.json')
    pix3d = json.load(open(pix3d_dir, 'r'))
    for cls in cls_all:
        in_path = os.path.join(root, '../input/img', cls)
        out_path = os.path.join(root, '../output', cls, 'images_crop_object')
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        # images = os.path.join(root, 'images')
        # images_crop_object = os.path.join(root, 'images_crop_object')
        box2d_all = get_box2d(pix3d, cls)
        crop_and_resize(in_path, out_path, box2d_all)


if __name__ == '__main__':
    crop_and_resize_all()
