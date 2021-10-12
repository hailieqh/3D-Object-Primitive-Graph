import scipy.io
import numpy as np
import os
import random
import json
import pdb


def check_image_voxel_match(cls):
    root = os.path.abspath('.')
    out_dir = os.path.join(root, '../output', cls)
    # out_dir = '/Users/heqian/Research/projects/primitive-based_3d/data/all_classes/chair'
    voxel_txt_dir = os.path.join(out_dir, 'voxeltxt')
    voxel_dirs = {x: os.path.join(voxel_txt_dir, 'voxel_{}.txt'.format(x))
                  for x in ['train', 'val', 'test']}
    img_dirs = {x: os.path.join(voxel_txt_dir, '{}.txt'.format(x))
                  for x in ['train', 'val', 'test']}
    with open(os.path.join(out_dir, 'voxels_dir_{}.txt'.format(cls)), 'r') as f:
        voxel_all = f.readlines()
    voxel_names = {}
    img_names = {}
    for phase in ['train', 'val', 'test']:
        with open(os.path.join(voxel_dirs[phase]), 'r') as f:
            voxel_names[phase] = f.readlines()
        with open(os.path.join(img_dirs[phase]), 'r') as f:
            img_names[phase] = f.readlines()
    # pix3d_dir = os.path.join(root, '../input/pix3d.json')
    # pix3d = json.load(open(pix3d_dir, 'r'))
    match_id = scipy.io.loadmat(os.path.join(out_dir, 'img_voxel_idxs.mat'))
    img_match_vox = {x: [] for x in ['train', 'val', 'test']}
    for phase in ['train', 'val', 'test']:
        for img in img_names[phase]:
            id_img_ori = int(img.split('.')[0]) # 1-3839
            img_id_real = list(match_id['img_idxs'][0]).index(id_img_ori)  # 0-3493
            voxel_id_ori = match_id['voxel_idxs'][0, img_id_real]  # 1-216
            vox = voxel_all[voxel_id_ori - 1]
            img_match_vox[phase].append(vox)
            # img_match_vox[phase].append('model/'+vox)
    img_match_vox = {x: sorted(set(img_match_vox[x])) for x in ['train', 'val', 'test']}
    # pdb.set_trace()
    for phase in ['train', 'val', 'test']:
        if len(set(voxel_names[phase]).difference(set(img_match_vox[phase]))) > 0:
            print('error')
        if len(set(img_match_vox[phase]).difference(set(voxel_names[phase]))) > 0:
            print('error')
        for name in voxel_names[phase]:
            if name not in img_match_vox[phase]:
                print(name)
        for name in img_match_vox[phase]:
            if name not in voxel_names[phase]:
                print(name)


def split_voxel_then_image(cls):
    # data_dir = '/Users/heqian/Research/projects/3dprnn/data/pix3d'
    split_by_model = True # True-split by 216 models, False-split by 34 images
    ##  split voxels into train, val, test
    root = os.path.abspath('.')
    out_dir = os.path.join(root, '../output', cls)
    voxel_txt_dir = os.path.join(out_dir, 'voxeltxt')
    if not os.path.exists(voxel_txt_dir):
        os.makedirs(voxel_txt_dir)
    voxel_train_txtpath = os.path.join(voxel_txt_dir, 'voxel_train.txt')
    voxel_val_txtpath = os.path.join(voxel_txt_dir, 'voxel_val.txt')
    voxel_test_txtpath = os.path.join(voxel_txt_dir, 'voxel_test.txt')
    voxel_ftrain = open(voxel_train_txtpath, 'w')
    voxel_fval = open(voxel_val_txtpath, 'w')
    voxel_ftest = open(voxel_test_txtpath, 'w')
    voxel_ltrain = []
    voxel_lval = []
    voxel_ltest = []
    voxel_ctrain = 0
    voxel_cval = 0
    voxel_ctest = 0
    with open(os.path.join(out_dir, 'voxels_dir_{}.txt'.format(cls)), 'r') as f:
        voxel_dirs = f.readlines()
    for i in range(len(voxel_dirs)):
        voxel_dirs[i] = voxel_dirs[i].strip()
        voxel_dirs[i] = voxel_dirs[i]
        tmp = random.random()
        if tmp < 0.65:
            voxel_ftrain.write(voxel_dirs[i]+'\n')
            voxel_ltrain.append(voxel_dirs[i])
            voxel_ctrain += 1
        elif tmp >= 0.65 and tmp < 0.8:
            voxel_fval.write(voxel_dirs[i]+'\n')
            voxel_lval.append(voxel_dirs[i])
            voxel_cval += 1
        else:
            voxel_ftest.write(voxel_dirs[i]+'\n')
            voxel_ltest.append(voxel_dirs[i])
            voxel_ctest += 1
    voxel_ftrain.close()
    voxel_fval.close()
    voxel_ftest.close()

    ##  split images into train, val, test, according to voxels
    # img_voxel_idxs = []
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
    pix3d_dir = os.path.join(root, '../input/pix3d.json')
    pix3d = json.load(open(pix3d_dir, 'r'))
    for i in range(len(pix3d)):
        # if json_file[i]['img'][4:9] == 'chair' and json_file[i]['voxel'] not in voxel_dirs:
        #   print(json_file[i]['img'], json_file[i]['voxel'])
        voxel_dir = pix3d[i]['voxel'][6:]
        if voxel_dir in voxel_dirs:
            # pdb.set_trace()
            img_file = pix3d[i]['img'].split('/')[-1]  #[10:]
            img_id = int(img_file.split('.')[0])    #int(pix3d[i]['img'][10:14])
            img_idxs.append(img_id)
            voxel_idxs.append(voxel_dirs.index(voxel_dir) + 1)
            # img_voxel_idxs.append(voxel_dirs.index(voxel_dir))
            # if img_id != len(img_voxel_idxs):
            #     print('Error!!!=======', img_id)
            if split_by_model:
                if voxel_dir in voxel_ltrain:
                    ftrain.write(img_file+'\n')
                    ctrain += 1
                elif voxel_dir in voxel_lval:
                    fval.write(img_file+'\n')
                    cval += 1
                elif voxel_dir in voxel_ltest:
                    ftest.write(img_file+'\n')
                    ctest += 1
            else:
                tmp = random.random()
                if tmp < 0.65:
                    ftrain.write(img_file+'\n')
                    ctrain += 1
                elif tmp >= 0.65 and tmp < 0.8:
                    fval.write(img_file+'\n')
                    cval += 1
                else:
                    ftest.write(img_file+'\n')
                    ctest += 1
    ftrain.close()
    fval.close()
    ftest.close()
    # scipy.io.savemat(os.path.join(out_dir, 'img_voxel_idxs.mat'),
    #                  {'img_voxel_idxs': np.array(img_voxel_idxs)})
    scipy.io.savemat(os.path.join(out_dir, 'img_voxel_idxs.mat'),
                     {'img_idxs': np.array(img_idxs), 'voxel_idxs': np.array(voxel_idxs)})

    print(voxel_ctrain+voxel_cval+voxel_ctest, voxel_ctrain, voxel_cval, voxel_ctest)
    print(ctrain+cval+ctest, ctrain, cval, ctest)
    print(len(img_idxs))


if __name__ == '__main__':
    cls_all = ['chair', 'bed', 'bookcase', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
    cls = 'table'
    # for cls in cls_all:
    split_voxel_then_image(cls)
    check_image_voxel_match(cls)
