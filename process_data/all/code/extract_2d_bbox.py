import os
import numpy as np
import json
import torch
import scipy.io
import cv2
import copy
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import ipdb


def load_prim_points_2d(data_dir, obj_class):
    labels = load_prim_points_2d_one_class(data_dir)
    return labels


def load_prim_points_2d_one_class(data_dir):
    kp_file = 'kp_2d_projected{}.json'.format(full_model)
    file_path = os.path.join(data_dir, kp_file)
    labels = json.load(open(file_path, 'r'))
    return labels


def get_prim_box_2d(item, match_id, id_img_ori, prim_points_2d, max_length, eval=False, inverse=False):
    img_id_real = list(match_id['img_idxs'][0]).index(id_img_ori)  # 0-3493
    label_kp = np.array(prim_points_2d['after'][str(img_id_real)]['keypoint'])
    if int(prim_points_2d['after'][str(img_id_real)]['name'][:4]) != id_img_ori:
        print(item, prim_points_2d['after'][str(img_id_real)]['name'], str(id_img_ori) + '.png')
    prim_box_2d = prim_points_to_box_2d(label_kp, max_length, eval, inverse=inverse)
    # if scale_box2d > 1.:##1.1
    #     prim_box_2d = enlarge_box2d(prim_box_2d)
    prim_box_2d = torch.transpose(torch.Tensor(prim_box_2d), 0, 1)  # (4, max_length)
    # prim_box_2d /= opt.input_res##224
    if True:
        length = prim_box_2d.size(1)
        prim_box_2d_new = np.zeros((4, length//3))
        for i in range(0, length, 3):
            prim_box_2d_new[:, i//3] = prim_box_2d[:, i]
        prim_box_2d = prim_box_2d_new
    return prim_box_2d.transpose()


def prim_points_to_box_2d(label_kp, max_length, eval=False, inverse=False):
    prim_num = label_kp.shape[0] // 8
    prim_box_2d = np.zeros((max_length, 4))
    for i in range(prim_num):
        j = i
        if inverse and not eval:
            j = prim_num - 1 - i
        prim_kp_i = label_kp[i*8 : i*8+8]
        min_x, min_y = np.min(prim_kp_i, axis=0)
        max_x, max_y = np.max(prim_kp_i, axis=0)
        prim_box_2d[j * 3:j * 3 + 3, :] = np.array([min_x, min_y, max_x, max_y])
        # prim_box_2d[i, :] = np.array([min_x, min_y, max_x, max_y])
    return prim_box_2d[:prim_num*3]


def load_model_id(root_dir, obj_class):
    model_id = scipy.io.loadmat(os.path.join(root_dir, 'img_voxel_idxs.mat'))  # 1-3839, 1-216
    return model_id


def get_model_id(id_img_ori, match_id): # 1-3839
    img_id_real = list(match_id['img_idxs'][0]).index(id_img_ori)  # 0-3493
    voxel_id_ori = match_id['voxel_idxs'][0, img_id_real]  # 1-216
    return voxel_id_ori - 1


def load_names(phase, root_dir, obj_class):
    image_names = load_names_one_class(phase, root_dir, obj_class)
    return image_names


def load_names_one_class(phase, root_dir, obj_class):
    if phase == 'train_val':
        image_names_train, lines_train = load_names_one('train', root_dir, obj_class)
        image_names_val, lines_val = load_names_one('val', root_dir, obj_class)
        image_names = image_names_train + image_names_val
        lines = lines_train + lines_val
    else:
        image_names, lines = load_names_one(phase, root_dir, obj_class)
    return image_names


def load_names_one(phase, root_dir, obj_class):
    down_sample = False
    names_file_dir = root_dir + '/voxeltxt/' + phase + '.txt'
    # names_file_dir = get_names_dir(phase, root_dir, obj_class)
    f = open(names_file_dir, 'r')
    lines = f.readlines()
    image_names = []
    count = 0
    for line in lines:
        # idx = int(line[0:4])
        if down_sample:
            if count % 10 == 0:
                image_names.append(line[:-1])
        else:
            image_names.append(line[:-1])
        count += 1
    return image_names, lines


def get_names_dir(phase, root_dir, obj_class):## txt names, you can set it
    if phase[:4] == 'test':
        if 'ts' in opt.test_version or opt.test_version == '1':
            phase = 'test'
        if opt.test_version == 'train':
            phase = 'train'
            # down_sample = True
    if phase == 'val' and opt.train_val and obj_class == 'chair':
        phase = 'val_429'
    if opt.model_split:
        phase_file_dir = root_dir + '/voxeltxt/' + phase + '.txt'
    else:
        phase_file_dir = root_dir + '/' + phase + '.txt'
    return phase_file_dir


def load_image(image_name, root_dir):
    if True:
        image_path = os.path.join(root_dir, 'images_crop_object', image_name[0:4] + '.png')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # # image_path = os.path.join(self.root_dir, 'images', image_name)
        # if image_name[-4:] == '.png': #if image_name == '2905.png':
        #     image = io.imread(image_path)
        # else:
        #     image = plt.imread(image_path) # np.float32 0-1
        #     # if image_name[-4:] == '.png':
        #     #   image = np.uint8(image*255) # io.imread np.uint8 0-255
        image = check_image_channel(image, image_name)
    return image


def check_image_channel(image, image_name):
    if len(image.shape) == 2:
        image = np.array([image, image, image]).transpose((1, 2, 0)) # gray image
    if len(image.shape) < 3:
        print(image_name)
    if image.shape[2] == 4:
        image = copy.deepcopy(image[:, :, 0:3])
    return image


def visualize_img(img_name, image, label, type=None):
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.imshow(image)
    if label is not None:
        if 'after_prim' in type:
            label, box2d = label
            num = box2d.shape[0]    # // 2
            min_xy = box2d[:, :2]   #box2d[:num, :]
            max_xy = box2d[:, 2:]   #box2d[num:, :]
            for box_i in range(num):
                if min_xy[box_i, 0] != max_xy[box_i, 0] and min_xy[box_i, 1] != max_xy[box_i, 1]:
                    # rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
                    rect = patches.Rectangle((min_xy[box_i, 0], min_xy[box_i, 1]),
                                             max_xy[box_i, 0] - min_xy[box_i, 0],
                                             max_xy[box_i, 1] - min_xy[box_i, 1],
                                             linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)  # Add the patch to the Axes
        # else:
        # plt.scatter(label[0, 0], label[0, 1], s=100, marker='.', c='g')
        # plt.scatter(label[1, 0], label[1, 1], s=100, marker='.', c='r')
        # plt.scatter(label[2, 0], label[2, 1], s=100, marker='.', c='b')
        # plt.scatter(label[3:, 0], label[3:, 1], s=100, marker='.', c='r')
    save_dir = os.path.join(visual_dir_2d, type)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_name = os.path.join(save_dir, img_name.split('.')[0] + '.png')
    plt.savefig(save_name)
    plt.close('all')


if __name__ == '__main__':
    box_2d_all = {}
    max_length = 50  # maximum number of boxes
    obj_class = 'chair'
    full_model = '_full'## or ''
    proposal = '_proposal' ## or ''
    root_dir = '../input/faster_rcnn/chair'
    save_dir = os.path.join(root_dir, 'box_2d_all{}{}.json'.format(full_model, proposal))
    visual_dir_2d = '../visual/faster_rcnn/chair'
    phases = ['train', 'val', 'test']
    match_id = load_model_id(root_dir, obj_class)
    prim_points_2d = load_prim_points_2d(root_dir, obj_class)
    max_proposal = 0

    for phase in phases:
        image_names = load_names(phase, root_dir, obj_class)
        if 'proposal' in proposal:
            proposal_dir = os.path.join(root_dir, 'results', '{}_box.json'.format(phase))
            box_proposals = json.load(open(proposal_dir, 'r'))
        for item in range(len(image_names)):
            image_name = image_names[item]
            if 'proposal' in proposal:
                prim_box_2d = box_proposals[str(item)]
                box_2d_all[image_name] = prim_box_2d
                max_proposal = max(max_proposal, len(prim_box_2d))
                prim_box_2d = np.array(prim_box_2d)
            else:
                id_img_ori = int(image_name[0:4])  # 1-3839
                prim_box_2d = get_prim_box_2d(item, match_id, id_img_ori, prim_points_2d, max_length)
                box_2d_all[image_name] = prim_box_2d.tolist()
            image = load_image(image_name, root_dir)
            # visualize_img(image_name, image, (None, prim_box_2d), 'after_prim{}{}'.format(full_model, proposal))
            print(phase, item, image_name)
    print(max_proposal)

    with open(save_dir, 'w') as f:
        json.dump(box_2d_all, f)
