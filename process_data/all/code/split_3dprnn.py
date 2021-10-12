import scipy.io
import numpy as np
import os
import random
import json
import pdb


def split_voxel_then_image(cls):
    # , 'depth_render_{}'.format(cls[7:])
    root = os.path.abspath('.')
    in_dir = os.path.join(root, '../input/3dprnn/depth_map')
    pre_match_id_file = os.path.join(in_dir, '../random_sample_id_mulfive.mat')
    pre_match_id = scipy.io.loadmat(pre_match_id_file)  # 5555, 1850, 0-888, 0-391, 0-199
    pre_match_id_crop = {}
    start_id = start_end_id[cls]['train'][0]
    end_id = start_end_id[cls]['train'][1] + 1
    pre_match_id_crop['train'] = list(pre_match_id['ind_train_mulfive'][0])[start_id:end_id]
    start_id = start_end_id[cls]['val'][0]
    end_id = start_end_id[cls]['val'][1] + 1
    pre_match_id_crop['val'] = list(pre_match_id['ind_val_mulfive'][0])[start_id:end_id]

    split_by_model = True # True-split by 216 models, False-split by 34 images
    out_dir = os.path.join(root, '../output', cls)
    voxel_txt_dir = os.path.join(out_dir, 'voxeltxt')
    if not os.path.exists(voxel_txt_dir):
        os.makedirs(voxel_txt_dir)
    f = open(os.path.join(out_dir, 'voxels_dir_{}.txt'.format(cls)), 'w')
    voxel_train_txtpath = os.path.join(voxel_txt_dir, 'voxel_train.txt')
    voxel_val_txtpath = os.path.join(voxel_txt_dir, 'voxel_val.txt')
    voxel_test_txtpath = os.path.join(voxel_txt_dir, 'voxel_test.txt')
    voxel_ftrain = open(voxel_train_txtpath, 'w')
    voxel_fval = open(voxel_val_txtpath, 'w')
    voxel_ftest = open(voxel_test_txtpath, 'w')
    voxel_ctrain = 0
    voxel_cval = 0
    voxel_ctest = 0
    img_idxs = []
    voxel_idxs = []
    train_txtpath = os.path.join(voxel_txt_dir, 'train.txt')
    val_txtpath = os.path.join(voxel_txt_dir, 'val.txt')
    test_txtpath = os.path.join(voxel_txt_dir, 'test.txt')
    ftrain = open(train_txtpath, 'w')
    fval = open(val_txtpath, 'w')
    ftest = open(test_txtpath, 'w')
    ctrain = 0
    cval = 0
    ctest = 0
    im_sum = (obj_num[cls]['train'] + obj_num[cls]['test']) * 5
    for i in range(im_sum):
        model_i = i // 5    # start from 0
        img_file = ('0000' + str(i + 1) + '.mat')[-8:]  # start from 1
        img_idxs.append(i + 1)
        voxel_idxs.append(model_i + 1)
        if i % 5 == 0:
            f.write(str(model_i) + '\n')
        if model_i in pre_match_id_crop['train']:
            if i % 5 == 0:
                voxel_ftrain.write(str(model_i) + '\n')
                voxel_ctrain += 1
            ftrain.write(img_file + '\n')
            ctrain += 1
        elif model_i in pre_match_id_crop['val']:
            if i % 5 == 0:
                voxel_fval.write(str(model_i) + '\n')
                voxel_cval += 1
            fval.write(img_file + '\n')
            cval += 1
        else:
            if i % 5 == 0:
                voxel_ftest.write(str(model_i) + '\n')
                voxel_ctest += 1
            ftest.write(img_file + '\n')
            ctest += 1
    voxel_ftrain.close()
    voxel_fval.close()
    voxel_ftest.close()
    ftrain.close()
    fval.close()
    ftest.close()
    f.close()
    scipy.io.savemat(os.path.join(out_dir, 'img_voxel_idxs.mat'),
                     {'img_idxs': np.array(img_idxs), 'voxel_idxs': np.array(voxel_idxs)})

    print(voxel_ctrain + voxel_cval + voxel_ctest, voxel_ctrain, voxel_cval, voxel_ctest)
    print(ctrain + cval + ctest, ctrain, cval, ctest)
    print(len(img_idxs))


if __name__ == '__main__':
    intervals = {'train': [3335, 4805, 5555], 'val': [1110, 1600, 1850]}
    start_end_id = {'3dprnnchair': {'train': [0, 3334], 'val': [0, 1109]},
                '3dprnntable': {'train': [3335, 4804], 'val': [1110, 1599]},
                '3dprnnnight_stand': {'train': [4805, 5554], 'val': [1600, 1849]}}
    cls_all = ['3dprnnchair', '3dprnntable', '3dprnnnight_stand']
    cls = '3dprnnnight_stand'
    obj_num = {'3dprnnchair': {}, '3dprnntable': {}, '3dprnnnight_stand': {}}
    obj_num['3dprnnnight_stand'] = {'train': 200, 'test': 86}
    obj_num['3dprnnchair'] = {'train': 889, 'test': 100}
    obj_num['3dprnntable'] = {'train': 392, 'test': 100}
    # for cls in cls_all:
    split_voxel_then_image(cls)
