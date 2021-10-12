import scipy.io
import numpy as np
import os
import random
import json
import pdb


def split(cls):
    # cls = 'chair'
    phase = 'train'
    root = os.path.abspath('.')
    out_dir = os.path.join(root, '../output', cls)
    voxel_txt_dir = os.path.join(out_dir, 'voxeltxt')
    voxel_dir = os.path.join(voxel_txt_dir, 'voxel_{}.txt'.format(phase))
    img_dir = os.path.join(voxel_txt_dir, '{}.txt'.format(phase))
    with open(os.path.join(out_dir, 'voxels_dir_{}.txt'.format(cls)), 'r') as f:
        voxel_all = f.readlines()
    with open(os.path.join(voxel_dir), 'r') as f:
        voxel_names = f.readlines()
    with open(os.path.join(img_dir), 'r') as f:
        img_names = f.readlines()
    match_id = scipy.io.loadmat(os.path.join(out_dir, 'img_voxel_idxs.mat'))
    img_train_stack = {'a':[], 'b':[]}
    for vox_name in voxel_names:
        # pdb.set_trace()
        voxel_id_ori = voxel_all.index(vox_name) + 1    # 1-216
        voxel_img_ids = []
        for i in range(match_id['voxel_idxs'].shape[1]):    # 0-3493
            if voxel_id_ori == match_id['voxel_idxs'][0, i]:
                id_img_ori = match_id['img_idxs'][0, i] # 1-3839
                voxel_img_ids.append(id_img_ori)
        if random.random() < 0.5:
            img_train_stack['a'] += voxel_img_ids
        else:
            img_train_stack['b'] += voxel_img_ids
    scipy.io.savemat(os.path.join(out_dir, 'stack_img_idxs.mat'),
        {'a': img_train_stack['a'], 'b': img_train_stack['b']})
    print(len(img_train_stack['a']))#, img_train_stack['a'])
    print(len(img_train_stack['b']))#, img_train_stack['b'])

    ## check
    img_id_all = []
    for name in img_names:
        id_img_ori = int(name.split('.')[0])
        img_id_all.append(id_img_ori)
    save_all = img_train_stack['a'] + img_train_stack['b']
    # pdb.set_trace()
    if len(save_all) != len(img_id_all):
        print('error')
    if len(set(save_all).difference(set(img_id_all))) > 0:
        print('error')
    if len(set(img_id_all).difference(set(save_all))) > 0:
        print('error')


if __name__ == '__main__':
    cls_all = ['chair', 'bed', 'bookcase', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
    cls = '3dprnnnight_stand'
    # for cls in cls_all:
    split(cls)
