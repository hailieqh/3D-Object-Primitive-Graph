import pdb
import cv2
import math
import json
import copy
import torch
import torchfile
import torch.nn as nn
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')

from lib.opts import *


def load_depth(phase, data_dir, intervals=None):
    # global opt
    # opt = get_opt()
    if phase in ['train', 'val']:
        # depth_tile train (5555*64*64) val (1850*64*64)
        file_name = os.path.join(data_dir, 'depth_map', opt.file_names['depth'][phase])
    elif phase[0:4] == 'test':
        # depth_tile(5000*64*64), match_id(100*2), randsampl(1*100), randsampl_x(1*100), randsampl_z(1*100)
        file_name = os.path.join(data_dir, 'depth_map', 'depth_mn_test_{}_ts.mat'.format(phase[5:]))  ##sublime
    else:
        raise NotImplementedError
    depth = scipy.io.loadmat(file_name)['depth_tile']
    if opt.composition is not None:
        depth = compose_depth(depth, intervals[phase])
    return depth


def compose_depth(in_depth, intervals):
    # import pdb
    # pdb.set_trace()
    num_chair, num_table, num_night_stand = interval_to_num(intervals)
    depth_chair = in_depth[:intervals[0], :, :]
    depth_table = in_depth[intervals[0]:intervals[1], :, :]
    depth_night_stand = in_depth[intervals[1]:, :, :]
    if opt.composition == 'balance':
        depth = np.zeros((num_chair * 3, 64, 64))
        depth[:num_chair, :, :] = depth_chair
        depth[num_chair:num_chair * 2, :, :] = np.concatenate((depth_table, depth_table,
                                                               depth_table[:(num_chair % num_table), :, :]), axis=0)
        depth[num_chair * 2:, :, :] = np.concatenate((depth_night_stand, depth_night_stand,
                                                      depth_night_stand, depth_night_stand,
                                                      depth_night_stand[:(num_chair % num_night_stand), :, :]), axis=0)
        return depth
    elif opt.composition == 'chair':
        return depth_chair
    elif opt.composition == 'table':
        return depth_table
    elif opt.composition == 'night_stand':
        return depth_night_stand
    else:
        raise NotImplementedError


def combine_depth(depth_train, depth_val, intervals_train, intervals_val):
    # if opt.composition in ['chair', 'table', 'night_stand']:
    if opt.composition in opt.file_names['obj_classes']:
        depth = np.concatenate((depth_train, depth_val), axis=0)
        return depth
    elif opt.composition == 'balance':
        intervals_train, intervals_val = update_intervals(intervals_train, intervals_val)
    depth = np.zeros((depth_train.shape[0]+depth_val.shape[0], 64, 64))
    start_id = 0
    end_id = intervals_train[0]
    depth[start_id:end_id, :, :] = copy.deepcopy(depth_train[:intervals_train[0], :, :])
    start_id = end_id
    end_id += intervals_val[0]
    depth[start_id:end_id, :, :] = copy.deepcopy(depth_val[:intervals_val[0], :, :])
    start_id = end_id
    end_id += intervals_train[1] - intervals_train[0]
    depth[start_id:end_id, :, :] = copy.deepcopy(depth_train[intervals_train[0]:intervals_train[1], :, :])
    start_id = end_id
    end_id += intervals_val[1] - intervals_val[0]
    depth[start_id:end_id, :, :] = copy.deepcopy(depth_val[intervals_val[0]:intervals_val[1], :, :])
    start_id = end_id
    end_id += depth_train.shape[0] - intervals_train[1]
    depth[start_id:end_id, :, :] = copy.deepcopy(depth_train[intervals_train[1]:, :, :])
    start_id = end_id
    end_id += depth_val.shape[0] - intervals_val[1]
    depth[start_id:end_id, :, :] = copy.deepcopy(depth_val[intervals_val[1]:, :, :])
    # import pdb
    # pdb.set_trace()
    return depth


def normalize_prims_wrt_one_prim(labels):
    ref_id = int(opt.prim_norm) # 0-n, No. primitive
    # file_name = opt.file_names['mean_std'][opt.obj_class]
    # test_init = scipy.io.loadmat(os.path.join(opt.data_dir, file_name))
    test_init = opt.mean_std
    # mean_x, mean_y, mean_r = test_init['mean_x'], test_init['mean_y'], test_init['mean_r']
    # std_x, std_y, std_r = test_init['std_x'], test_init['std_y'], test_init['std_r']
    label_statistics = {'t_x':[], 't_y':[], 't_z':[]}
    for i in range(len(labels)):
        label = labels[i]
        scale = np.array(label[b'x_vals']) / opt.v_size
        translation = np.array(label[b'y_vals']) / opt.v_size
        ref_s = copy.deepcopy(scale[ref_id * 3:ref_id * 3 + 3])
        ref_s = ref_s[2]  ## wrt height
        scale /= ref_s
        translation /= ref_s
        ref_s = copy.deepcopy(scale[ref_id * 3:ref_id * 3 + 3])
        ref_t = copy.deepcopy(translation[ref_id * 3:ref_id * 3 + 3])
        ref_t += ref_s / 2
        ins_length = len(scale)
        for j in range(0, ins_length, 3):
            translation[j] -= ref_t[0]
            translation[j + 1] -= ref_t[1]
            translation[j + 2] -= ref_t[2]
            label_statistics['t_x'].append(translation[j])
            label_statistics['t_y'].append(translation[j + 1])
            label_statistics['t_z'].append(translation[j + 2])
        label[b'x_vals'] = list(scale)
        label[b'y_vals'] = list(translation)
        labels[i] = label
    return labels


def denormalize_prims(labels):
    # file_name = opt.file_names['mean_std'][opt.obj_class]
    # test_init = scipy.io.loadmat(os.path.join(opt.data_dir, file_name))
    test_init = opt.mean_std
    mean_x, mean_y, mean_r = test_init['mean_x'], test_init['mean_y'], test_init['mean_r']
    std_x, std_y, std_r = test_init['std_x'], test_init['std_y'], test_init['std_r']
    for i in range(len(labels)):
        label = labels[i]
        scale = np.array(label[b'x_vals'])
        translation = np.array(label[b'y_vals'])
        rotation = np.array(label[b'r_vals'])
        scale = scale * std_x[0, 0] + mean_x[0, 0]
        translation = translation * std_y[0, 0] + mean_y[0, 0]
        rotation = rotation * std_r[0, 0] + mean_r[0, 0]
        label[b'x_vals'] = list(scale)
        label[b'y_vals'] = list(translation)
        label[b'r_vals'] = list(rotation)
        # print(label)
        labels[i] = label
    return labels


def load_primset_to_t7_format(data_dir, obj_class):
    global opt
    opt = get_opt()
    if 'all' in obj_class:
        labels = {}
        mean_std = {}
        for cls in opt.file_names['obj_classes']:
            tmp_dir = os.path.join(data_dir, cls)
            labels_cls, mean_std_cls = load_primset_to_t7_format_one_class(tmp_dir)
            labels[cls] = labels_cls
            mean_std[cls] = mean_std_cls
    else:
        labels, mean_std = load_primset_to_t7_format_one_class(data_dir)
    opt.mean_std = mean_std
    return labels


def load_primset_to_t7_format_one_class(data_dir):
    # pdb.set_trace()
    primset = scipy.io.loadmat(os.path.join(data_dir, opt.file_names['primset']))
    primset = primset['primset']
    num = primset.shape[0]
    labels = []
    mean_std = {'mean_x':0, 'mean_y':0, 'mean_r':0, 'std_x':0, 'std_y':0, 'std_r':0}
    length = 0
    for i in range(num):
        label = {}
        label[b'x_vals'] = []
        label[b'y_vals'] = []
        label[b'e_vals'] = []
        label[b'r_vals'] = []
        label[b'rt_vals'] = []
        label[b'rs_vals'] = []
        label[b'cls'] = []
        prims = primset[i, 0]
        ori = prims['ori'][0, 0]
        sym = prims['sym'][0, 0]
        cls = prims['cls'][0, 0][0] # start from 1 (will be decreased later in get_label)
        if opt.full_model:
            prim_all, cls_all = combine_ori_sym(ori, sym, cls)
        else:
            prim_all, cls_all = ori, cls
        for j in range(prim_all.shape[0]):
            label[b'x_vals'] += prim_all[j, 10:13].tolist()
            label[b'y_vals'] += prim_all[j, 13:16].tolist()
            label[b'e_vals'] += [0, 0, 0]
            label[b'r_vals'] += (prim_all[j, 16:19] * prim_all[j, 19]).tolist()
            label[b'rt_vals'] += [prim_all[j, 19], prim_all[j, 19], prim_all[j, 19]]
            # if abs(ori[j, 19]) > math.pi/4:
            #     pdb.set_trace()
            label[b'rs_vals'] += prim_all[j, 16:19].tolist()
            label[b'cls'] += [cls_all[j], cls_all[j], cls_all[j]]
        label[b'e_vals'][-1] = 1
        labels.append(label)
        length += len(label[b'x_vals'])
        mean_std['mean_x'] += sum(label[b'x_vals'])
        mean_std['mean_y'] += sum(label[b'y_vals'])
        if opt.norm_rot == 'theta':
            mean_std['mean_r'] += sum(label[b'rt_vals'])
        elif opt.norm_rot == 'all':
            mean_std['mean_r'] += sum(label[b'r_vals'])
        else:
            raise NotImplementedError
    if opt.norm_st == 'st':
        mean_st = (mean_std['mean_x'] + mean_std['mean_y']) / length / 2.
        mean_std['mean_x'] = mean_st
        mean_std['mean_y'] = mean_st
    else:
        mean_std['mean_x'] /= length
        mean_std['mean_y'] /= length
    mean_std['mean_r'] /= length
    mean_std = compute_std(labels, length, mean_std)
    labels = normalize_t7(labels, mean_std)
    for key, value in mean_std.items():
        mean_std[key] = np.array([[value]])
    # if opt.tmp:
    #     check_t7_mean_std(labels, data_dir)
    return labels, mean_std


def combine_ori_sym(ori, sym, cls):
    prim_all = copy.deepcopy(ori)
    cls_all = copy.deepcopy(cls)
    for i in range(ori.shape[0]):
        if np.sum(np.abs(sym[i, 10:])) > 0:
            prim_all = np.vstack((prim_all, sym[i:i+1, :]))
            cls_all = np.hstack((cls_all, cls[i:i+1]))
    return prim_all, cls_all


def compute_std(labels, length, mean_std):
    for label in labels:
        mean_std['std_x'] += np.sum((np.array(label[b'x_vals']) - mean_std['mean_x']) ** 2)
        mean_std['std_y'] += np.sum((np.array(label[b'y_vals']) - mean_std['mean_y']) ** 2)
        if opt.norm_rot == 'theta':
            mean_std['std_r'] += np.sum((np.array(label[b'rt_vals']) - mean_std['mean_r']) ** 2)
        elif opt.norm_rot == 'all':
            mean_std['std_r'] += np.sum((np.array(label[b'r_vals']) - mean_std['mean_r']) ** 2)
        else:
            raise NotImplementedError
    if opt.norm_st == 'st':
        std_st = np.sqrt((mean_std['std_x'] + mean_std['std_y']) / length / 2.)
        mean_std['std_x'] = std_st
        mean_std['std_y'] = std_st
    else:
        mean_std['std_x'] = np.sqrt(mean_std['std_x'] / length)
        mean_std['std_y'] = np.sqrt(mean_std['std_y'] / length)
    mean_std['std_r'] = np.sqrt(mean_std['std_r'] / length)
    return mean_std


def normalize_t7(labels, mean_std):
    for label in labels:
        label[b'x_vals'] = ((np.array(label[b'x_vals']) - mean_std['mean_x']) / mean_std['std_x']).tolist()
        label[b'y_vals'] = ((np.array(label[b'y_vals']) - mean_std['mean_y']) / mean_std['std_y']).tolist()
        label[b'r_vals'] = ((np.array(label[b'r_vals']) - mean_std['mean_r']) / mean_std['std_r']).tolist()
    return labels


def check_t7_mean_std(labels, data_dir):
    pdb.set_trace()
    file_name = opt.file_names['mean_std'][opt.obj_class]
    test_init = scipy.io.loadmat(os.path.join(data_dir, file_name))
    for key, value in opt.mean_std.items():
        if abs(opt.mean_std[key][0, 0] - test_init[key][0, 0]) > 1e-8:
            print(key, opt.mean_std[key])
            print(key, test_init[key])
    labels_saved = load_labels_from_t7('train', data_dir)
    assert len(labels) == len(labels_saved)
    for i in range(len(labels)):
        label = labels[i]
        label_saved = labels_saved[i]
        for key, value in label.items():
            if key not in label_saved.keys():
                continue
            for j in range(len(value)):
                if abs(label[key][j] - label_saved[key][j]) > 1e-8:
                    pdb.set_trace()
                    print(i, key, j)
                    print(labels[i][key][j])
                    print(labels_saved[i][key][j])
                    break


def load_labels_from_t7(phase, root_dir=None, intervals=None):
    # if root_dir is None:
    #     root_dir = opt.data_dir
    file_name = opt.file_names['norm_label']
    if file_name[-3:] == 'pth':
        labels = torch.load(os.path.join(root_dir, file_name))  # list - dict -list
    else:
        labels = torchfile.load(os.path.join(root_dir, file_name))  # list - dict -list
    if opt.composition is not None:
        labels = compose_labels(labels, intervals[phase])
    if opt.global_denorm:
        labels = denormalize_prims(labels)
    # print('before', labels[0])
    if opt.prim_norm is not None:
        labels = normalize_prims_wrt_one_prim(labels)
    # print('after', labels[0])
    return labels


def plot_gt_distribution(label):
    # plot scale-translation two dimensional distribution (scattering points)
    save_name = '//'
    fig = plt.figure()
    plt.scatter(label[:, 0], label[:, 1], s=100, marker='.', c='r')
    plt.savefig(save_name)
    plt.close('all')


def compose_labels(in_labels, intervals):
    # import pdb
    # pdb.set_trace()
    num_chair, num_table, num_night_stand = interval_to_num(intervals)  # train (3335, 1470, 750) val (1110, 490, 250)
    labels_chair = in_labels[:intervals[0]]
    labels_table = in_labels[intervals[0]:intervals[1]]
    labels_night_stand = in_labels[intervals[1]:]
    if opt.composition == 'balance':
        labels = labels_chair
        labels += labels_table * (num_chair // num_table) + labels_table[:(num_chair % num_table)]
        labels += labels_night_stand * (num_chair // num_night_stand) + labels_night_stand[:(num_chair % num_night_stand)]
        return labels
    elif opt.composition == 'chair':
        return labels_chair
    elif opt.composition == 'table':
        return labels_table
    elif opt.composition == 'night_stand':
        return labels_night_stand
    else:
        raise NotImplementedError


def combine_labels(labels_train, labels_val, intervals_train, intervals_val):
    # import pdb
    # pdb.set_trace()
    # if opt.composition in ['chair', 'table', 'night_stand']:
    if opt.composition in opt.file_names['obj_classes']:
        labels = labels_train + labels_val
        return labels
    elif opt.composition == 'balance':
        intervals_train, intervals_val = update_intervals(intervals_train, intervals_val)
    labels = []
    labels += labels_train[:intervals_train[0]] + labels_val[:intervals_val[0]]
    labels += labels_train[intervals_train[0]:intervals_train[1]] + labels_val[intervals_val[0]:intervals_val[1]]
    labels += labels_train[intervals_train[1]:] + labels_val[intervals_val[1]:]
    return labels


def update_intervals(intervals_train, intervals_val):
    intervals_train = [intervals_train[0], intervals_train[0] * 2, intervals_train[0] * 3]
    intervals_val = [intervals_val[0], intervals_val[0] * 2, intervals_val[0] * 3]
    return intervals_train, intervals_val


def interval_to_num(intervals):
    num_chair = intervals[0]
    num_table = intervals[1] - intervals[0]
    num_night_stand = intervals[2] - intervals[1]
    return num_chair, num_table, num_night_stand


def load_test_init(obj_class=None):
    # if obj_class is None:
    #     obj_class = opt.obj_class
    if opt.init_source == 'given' or opt.save_gt_t7:
        # mean_x, mean_y, mean_r, std_x, std_y, std_r, test_sample(5*777), test_ret_num(777*1)
        # file_name = os.path.join(opt.data_dir, 'sample_generation/test_NNfeat_mn_{}.mat'.format(obj_class))
        file_name = opt.file_names['mean_std'][obj_class]
        test_init = opt.mean_std
    else:
        # mean_x, mean_y, mean_r, std_x, std_y, std_r, test_sample(5000*5), test_ret_num(5000*1)
        file_name = os.path.join(opt.init_source, 'test_NNfeat_mn_{}_l2.mat'.format(obj_class))
        test_init = scipy.io.loadmat(file_name)
    return test_init


def load_length(phase, root_dir, image_names, obj_class):
    len_dir = opt.len_dir
    if 'all' in obj_class:
        length_all = {}
        for cls in opt.file_names['obj_classes']:
            #tmp_dir = os.path.join(root_dir, cls)
            len_dir_cls = len_dir + '_' + cls
            length_all_cls = load_length_one_class(phase, root_dir, image_names[cls], cls, len_dir_cls, obj_class)
            length_all[cls] = length_all_cls
    else:
        length_all = load_length_one_class(phase, root_dir, image_names, obj_class, len_dir)
    return length_all


def load_length_one_class(phase, root_dir, image_names, obj_class, len_dir, obj_class_opt=' '):
    if phase == 'train_val':
        tmp_dir = root_dir
        if 'all' in obj_class_opt:
            tmp_dir = os.path.join(root_dir, obj_class)
        image_names_train, _ = load_names_one('train', tmp_dir, obj_class)
        image_names_val, _ = load_names_one('val', tmp_dir, obj_class)
        length_all_train = load_length_one('train', root_dir, image_names_train, obj_class, len_dir)
        length_all_val = load_length_one('val', root_dir, image_names_val, obj_class, len_dir)
        length_all = np.vstack((length_all_train, length_all_val))
    else:
        length_all = load_length_one(phase, root_dir, image_names, obj_class, len_dir)
    return length_all


def load_length_one(phase, root_dir, image_names, obj_class, len_dir):
    # len_source: None -- maintain gt train val and pred test, return all 0
    # len_source: stack -- stack train and pred val test, return
    # test is not from this function
    ## max length then prune
    gt_pred = 'pred'
    if opt.len_source == 'stack' and phase == 'train':
        stack_idxs_dir = '/stack_img_idxs.mat'
        if 'all' in opt.obj_class:
            stack_idxs_dir = '/' + obj_class + stack_idxs_dir
        stack_img_idxs = scipy.io.loadmat(root_dir + stack_idxs_dir)
        file_name_a = os.path.join(root_dir, len_dir, 'length_{}_val_a.mat'.format(gt_pred))
        file_name_b = os.path.join(root_dir, len_dir, 'length_{}_val_b.mat'.format(gt_pred))
        length_a = scipy.io.loadmat(file_name_a)['length_{}'.format(gt_pred)]
        length_b = scipy.io.loadmat(file_name_b)['length_{}'.format(gt_pred)]
        length_all = np.zeros((len(image_names), 1))
        count_a, count_b = 0, 0
        for i in range(len(image_names)):
            name = image_names[i]
            id_img_ori = int(name.split('.')[0])
            length = -1
            if id_img_ori in list(stack_img_idxs['a'][0]):
                # idx = list(stack_img_idxs['a'][0]).index(id_img_ori)
                # length = length_b[idx, 0]   # switch a and b split
                length = length_b[count_b, 0]  # switch a and b split
                count_b += 1
            if id_img_ori in list(stack_img_idxs['b'][0]):
                # idx = list(stack_img_idxs['b'][0]).index(id_img_ori)
                # length = length_a[idx, 0]   # switch a and b split
                length = length_a[count_a, 0]  # switch a and b split
                count_a += 1
            length_all[i, 0] = length
    elif opt.len_source == 'stack' and phase == 'val':
        if obj_class == 'chair' and opt.train_val:
            phase = 'val_429'
        file_name = 'length_{}_{}.mat'.format(gt_pred, phase)
        file_name = os.path.join(root_dir, len_dir, file_name)
        length_all = scipy.io.loadmat(file_name)['length_{}'.format(gt_pred)]
    else:
        length_all = np.zeros((len(image_names), 1))
    if opt.len_max > 0:
        length_all = np.ones((len(image_names), 1)) * opt.len_max
    # if opt.obj_class == 'table':
    if opt.extra_len > 0:
        length_all = length_all * (length_all > 0) + opt.extra_len
    # if opt.lstm_prop == 'gmm':
    #     length_all = length_all * 3
    return length_all


def get_names_dir(phase, root_dir, obj_class):
    if opt.stack_learn != 'None':
        phase = 'train'
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


def load_names(phase, root_dir, obj_class):
    if 'all' in obj_class:
        image_names = {}
        opt.file_names['img_nums'][phase] = {}
        for cls in opt.file_names['obj_classes']:
            tmp_dir = os.path.join(root_dir, cls)
            image_names_cls = load_names_one_class(phase, tmp_dir, cls)
            image_names[cls] = image_names_cls
            opt.file_names['img_nums'][phase][cls] = len(image_names_cls)
    else:
        image_names = load_names_one_class(phase, root_dir, obj_class)
    return image_names


def load_names_one_class(phase, root_dir, obj_class):
    global opt
    opt = get_opt()
    if phase == 'train_val':
        image_names_train, lines_train = load_names_one('train', root_dir, obj_class)
        image_names_val, lines_val = load_names_one('val', root_dir, obj_class)
        image_names = image_names_train + image_names_val
        lines = lines_train + lines_val
    else:
        image_names, lines = load_names_one(phase, root_dir, obj_class)
    if opt.stack_learn != 'None':
        image_names = stack_names(phase, root_dir, image_names)
        print(phase, len(lines), opt.stack_learn, len(image_names))
    return image_names


def load_names_one(phase, root_dir, obj_class):
    down_sample = False
    names_file_dir = get_names_dir(phase, root_dir, obj_class)
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


def stack_names(phase, root_dir, image_names):
    stack_img_idxs = scipy.io.loadmat(root_dir + '/stack_img_idxs.mat')
    image_names_new = []
    for name in image_names:
        id_img_ori = int(name.split('.')[0])
        if (opt.stack_learn == 'a' and phase == 'train') or \
                (opt.stack_learn == 'b' and phase == 'val'):
            if id_img_ori in list(stack_img_idxs['a'][0]):
                image_names_new.append(name)
        if (opt.stack_learn == 'b' and phase == 'train') or \
                (opt.stack_learn == 'a' and phase == 'val'):
            if id_img_ori in list(stack_img_idxs['b'][0]):
                image_names_new.append(name)
    image_names = image_names_new
    return image_names


def load_model_id(root_dir, obj_class):
    if 'all' in obj_class:
        model_id = {}
        for cls in opt.file_names['obj_classes']:
            tmp_dir = os.path.join(root_dir, cls)
            model_id_cls = scipy.io.loadmat(os.path.join(tmp_dir, 'img_voxel_idxs.mat'))  # 1-3839, 1-216
            model_id[cls] = model_id_cls
    else:
        model_id = scipy.io.loadmat(os.path.join(root_dir, 'img_voxel_idxs.mat'))  # 1-3839, 1-216
    return model_id


def get_model_id(id_img_ori, match_id): # 1-3839
    img_id_real = list(match_id['img_idxs'][0]).index(id_img_ori)  # 0-3493
    voxel_id_ori = match_id['voxel_idxs'][0, img_id_real]  # 1-216
    return voxel_id_ori - 1


def get_obj_class(item, phase):
    for cls in opt.file_names['obj_classes']:
        if item < opt.file_names['img_nums'][phase][cls]:
            return item, cls
        else:
            item -= opt.file_names['img_nums'][phase][cls]


def match_image_names_to_model_ids(phase, obj_class, root_dir):
    image_names = load_names(phase, root_dir, obj_class)
    match_id = load_model_id(root_dir, obj_class)
    model_ids = []
    for i in range(len(image_names)):
        id_img = int(image_names[i][0:4])
        voxel_id = get_model_id(id_img, match_id)
        model_ids.append(voxel_id)
    return model_ids


def load_image(image_name, root_dir):
    if image_name.split('.')[-1] == 'mat':
        image_path = os.path.join(root_dir, 'images_crop_object', image_name[0:4] + '.mat')
        image = scipy.io.loadmat(image_path)['depth'][:, :, np.newaxis]
        # print(image.shape, np.sum(image), type(image[0,0,0]))
        if '3dprnnnyu' in opt.obj_class:
            # print(np.min(image + (image == 0) * 1000))
            image = image - (np.min(image + (image == 0) * 1000) - 0.1) * (image > 0)
            # print(np.min(image), np.min(image + (image == 0) * 1000))
        if opt.encoder == 'resnet':
            image = check_image_channel(image[:, :, 0], image_name)
    else:
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
    # print('load', type(image[0,0,0]), np.max(image))
    return image


def check_image_channel(image, image_name):
    if len(image.shape) == 2:
        image = np.array([image, image, image]).transpose((1, 2, 0)) # gray image
    if len(image.shape) < 3:
        print(image_name)
    if image.shape[2] == 4:
        image = copy.deepcopy(image[:, :, 0:3])
    return image


def transform_image_label(id_img_ori, image, label=None, transform=None, prefix=''):
    if opt.visual_data:
        visualize_img(image, label, id_img_ori, transformed=False, prefix=prefix+'before')
    sample = {'image': image, 'label': label}
    if transform is not None:
        sample = transform(sample)
    if opt.visual_data:
        visualize_img(sample['image'], sample['label'], id_img_ori, transformed=True, prefix=prefix+'after')
    return sample


def visualize_img(image, label, i, transformed=False, prefix=''):
    if not transformed:
        save_name = 'out' + opt.env + '/{}{}.png'.format(prefix, i)
    else:
        image = image.numpy().transpose((1, 2, 0))
        if label is not None:
            if opt.loss_box2d is not None or opt.box2d_source is not None:
                label = label.numpy()
            else:
                label = label.numpy() * 256.0 / 64.0
        save_name = 'out' + opt.env + '/{}{}.png'.format(prefix, i)
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.imshow(image)
    if label is not None:
        if opt.loss_box2d is not None or opt.box2d_source is not None:
            num = label.shape[0] // 2
            min_xy = label[:num, :]
            max_xy = label[num:, :]
            for box_i in range(num):
                if min_xy[box_i, 0] != max_xy[box_i, 0] and min_xy[box_i, 1] != max_xy[box_i, 1]:
                    # rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
                    rect = patches.Rectangle((min_xy[box_i, 0], min_xy[box_i, 1]),
                                             max_xy[box_i, 0] - min_xy[box_i, 0],
                                             max_xy[box_i, 1] - min_xy[box_i, 1],
                                             linewidth=1, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)  # Add the patch to the Axes
        else:
            plt.scatter(label[:, 0], label[:, 1], s=100, marker='.', c='r')
    plt.savefig(save_name)
    plt.close('all')


def crop_bbox(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    image = copy.deepcopy(image[int(y_min):int(y_max), int(x_min):int(x_max), :])
    return image


def get_prim_box_2d(item, match_id, id_img_ori, prim_points_2d, max_length, image, transform, eval=False, inverse=False):
    # global opt
    # opt = get_opt()
    img_id_real = list(match_id['img_idxs'][0]).index(id_img_ori)  # 0-3493
    label_kp = np.array(prim_points_2d['after'][str(img_id_real)]['keypoint'])
    if int(prim_points_2d['after'][str(img_id_real)]['name'][:4]) != id_img_ori:
        print(item, prim_points_2d['after'][str(img_id_real)]['name'], str(id_img_ori) + '.png')
    prim_box_2d = prim_points_to_box_2d(label_kp, max_length, eval, inverse=inverse)
    prim_box_2d = np.concatenate((prim_box_2d[:, 0:2], prim_box_2d[:, 2:4]), axis=0)  # box to kp
    sample = transform_image_label(id_img_ori, image, label=prim_box_2d, transform=transform)
    prim_box_2d = sample['label']
    prim_box_2d = torch.cat((prim_box_2d[:max_length, :], prim_box_2d[max_length:, :]),
                            dim=1)  # kp to box
    if opt.scale_box2d > 1.:
        prim_box_2d = enlarge_box2d(prim_box_2d.numpy())
    prim_box_2d = torch.transpose(prim_box_2d, 0, 1)  # (4, max_length)
    prim_box_2d /= opt.input_res
    if opt.xyz == 3:
        # print('before box2d', prim_box_2d)
        length = prim_box_2d.size(1)
        prim_box_2d_new = np.zeros((4, length//3))
        for i in range(0, length, 3):
            prim_box_2d_new[:, i//3] = prim_box_2d[:, i]
        prim_box_2d = prim_box_2d_new
        # print('after box2d', prim_box_2d)
    else:
        prim_box_2d = prim_box_2d.numpy()
    return prim_box_2d, sample


def enlarge_box2d(prim_box_2d):
    times = opt.scale_box2d
    box_out = []
    for i in range(prim_box_2d.shape[0]):
        box_i = prim_box_2d[i]
        if box_i[0] == box_i[2] or box_i[1] == box_i[3]:
            box_out.append(box_i * 0)
            continue
        box_len = box_i[2:] - box_i[:2]
        margin = box_len * (times - 1) / 2
        box_i[:2] = np.maximum(box_i[:2] - margin, 0)
        box_i[2:] = np.minimum(box_i[2:] + margin, opt.input_res)
        box_out.append(box_i)
    # import pdb;pdb.set_trace()
    return torch.Tensor(box_out)


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
    return prim_box_2d


def load_prim_points_2d(data_dir, obj_class):
    if 'all' in obj_class:
        labels = {}
        for cls in opt.file_names['obj_classes']:
            tmp_dir = os.path.join(data_dir, cls)
            labels_cls = load_prim_points_2d_one_class(tmp_dir)
            labels[cls] = labels_cls
    else:
        labels = load_prim_points_2d_one_class(data_dir)
    return labels


def load_prim_points_2d_one_class(data_dir):
    kp_file = 'kp_2d_projected.json'
    if opt.full_model:
        kp_file = 'kp_2d_projected_full.json'
    file_path = os.path.join(data_dir, kp_file)
    labels = json.load(open(file_path, 'r'))
    return labels


def load_box_proposals(data_dir, obj_class):
    # file = 'delete/box_2d_all_full_proposal.json'
    file = 'box_2d_all_full_proposal.json'
    assert opt.full_model
    file_path = os.path.join(data_dir, file)
    labels = json.load(open(file_path, 'r'))
    return labels


def box_proposal_nms(box_proposal):
    scores = [x[-1] for x in box_proposal]
    idxs = torch.argsort(torch.Tensor(scores), descending=True)
    results = [box_proposal[i][:-1] for i in idxs[:opt.n_prop]]
    return results


def box_proposal_transform(box_proposal, id_img_ori, max_length, image, transform):
    prim_box_2d = np.zeros((max_length, 4))
    prim_box_2d[:len(box_proposal)] = np.array(box_proposal)
    prim_box_2d = np.concatenate((prim_box_2d[:, 0:2], prim_box_2d[:, 2:4]), axis=0)  # box to kp
    sample = transform_image_label(id_img_ori, image, label=prim_box_2d, transform=transform, prefix='prop')
    prim_box_2d = sample['label']
    prim_box_2d = torch.cat((prim_box_2d[:max_length, :], prim_box_2d[max_length:, :]),
                            dim=1)  # kp to box
    if opt.scale_box2d > 1.:
        prim_box_2d = enlarge_box2d(prim_box_2d.numpy())
    prim_box_2d = torch.transpose(prim_box_2d, 0, 1)  # (4, max_length)
    prim_box_2d /= opt.input_res
    prim_box_2d = prim_box_2d.numpy()
    return prim_box_2d


def match_2d_box(box_proposal, prim_box_2d, num, id=None):
    eps = 1e-4
    match_id = -torch.ones(1, opt.n_prop).long()
    for i in range(num):
        prop = (copy.deepcopy(box_proposal[:, i]) * opt.input_res).astype(int)
        prop_mask = np.zeros((opt.input_res, opt.input_res))
        prop_mask[prop[1]:prop[3], prop[0]:prop[2]] = 1
        iou = []
        for j in range(prim_box_2d.shape[1]):
            gt = (copy.deepcopy(prim_box_2d[:, j]) * opt.input_res).astype(int)
            gt_mask = np.zeros((opt.input_res, opt.input_res))
            gt_mask[gt[1]:gt[3], gt[0]:gt[2]] = 1
            iou_ij = np.sum((prop_mask + gt_mask) == 2) / (np.sum((prop_mask + gt_mask) > 0) + eps)
            iou.append(iou_ij)
        idx = np.argmax(np.array(iou))
        match_id[0, i] = int(idx)
    return match_id


def load_bbox_2d(bbox_file_2d, root_dir):
    bbox_dir_2d = os.path.join(root_dir, bbox_file_2d)
    bbox_2d = scipy.io.loadmat(bbox_dir_2d)
    bbox_2d = bbox_2d['bboxes']
    return bbox_2d


def load_bbox_3d(bbox_file_3d, root_dir):
    bbox_dir_3d = os.path.join(root_dir, bbox_file_3d)
    bbox_3d = scipy.io.loadmat(bbox_dir_3d)
    bbox_3d = bbox_3d['primset']
    return bbox_3d


def get_bbox_3d_ins(gt_instance):
    expand = int(opt.demo is None) * opt.expand
    # id_model = id_img - 1
    # gt_instance = bbox_3d[id_model,0]['sem'][0,0]
    assert gt_instance.shape == (opt.n_sem, 9)
    part = np.zeros(opt.n_sem * opt.n_para)
    existence = np.zeros(opt.n_sem)
    for i_sem in range(opt.n_sem):
        min_xyz = gt_instance[i_sem, 0:3] - 1000*(gt_instance[i_sem, 0:3]>opt.v_size)   # 32
        len_xyz = gt_instance[i_sem, 6:9] + 1000*(gt_instance[i_sem, 6:9]<0)
        assert (min_xyz >= 0).all() and (min_xyz <= opt.v_size).all()
        assert (len_xyz >= 0).all() and (len_xyz <= opt.v_size).all()
        i_exist_min = (gt_instance[i_sem, 0:3]>=0).all() and (gt_instance[i_sem, 0:3]<=opt.v_size).all()
        i_exist_len = (gt_instance[i_sem, 6:9]>=0).all() and (gt_instance[i_sem, 6:9]<=opt.v_size).all()
        i_exist = i_exist_min and i_exist_len
        if i_exist:
            i_expand = expand
            existence[i_sem] = 1
        else:
            i_expand = 0
            existence[i_sem] = 0
        part[i_sem*opt.n_para     : i_sem*opt.n_para + 3] = np.maximum(min_xyz-i_expand, 0)
        part[i_sem*opt.n_para + 3 : i_sem*opt.n_para + 6] = np.minimum(len_xyz+i_expand, opt.v_size)
    part = part / float(opt.v_size)
    return part, existence


def normalize_bbox(bbox, mean_std_init):
    bbox *= float(opt.v_size)
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
                #### but it is ok cus denormalize shares the same means and stds
                # bbox[i_sem * opt.n_para + j] = (bbox[i_sem * opt.n_para + j] - mean_x[0, 0]) / std_x[0, 0]
                # bbox[i_sem * opt.n_para + j + 3] = (bbox[i_sem * opt.n_para + j + 3] - mean_y[0, 0]) / std_y[0, 0]
                bbox[i_sem * opt.n_para + j] = (bbox[i_sem * opt.n_para + j] - mean_y[0, 0])\
                                               / std_y[0, 0]
                bbox[i_sem * opt.n_para + j + 3] = (bbox[i_sem * opt.n_para + j + 3] - mean_x[0, 0])\
                                                   / std_x[0, 0]
    return bbox


def get_inverse_label(label):
    label_new = {}
    for key in label.keys():
        tmp = copy.deepcopy(label[key])
        if key != b'e_vals':
            tmp.reverse()
        label_new[key] = tmp
    return label_new


def get_inverse_mat(input_mat, rot_mat, cls_mat, ins_length):
    input_mat[:, :ins_length] = input_mat[:, :ins_length].flip(1)
    rot_mat[:, :ins_length] = rot_mat[:, :ins_length].flip(1)
    cls_mat[:, :ins_length] = cls_mat[:, :ins_length].flip(1)
    return input_mat, rot_mat, cls_mat


def get_label(item, labels, max_length, init=None, test_init=None, ins_id=None, prim_box_2d=None, inverse=False):
    if item >= 0:
        label = labels[item]
    else:
        label = labels
    ins_length = len(label[b'x_vals'])
    input_mat = torch.zeros(3, max_length)
    rot_mat = torch.zeros(2, max_length)
    cls_mat = torch.zeros(1, max_length)   #, dtype=torch.int32)
    if not opt.sigma_share:
        y_mask_mat = torch.zeros(1 + 6 * opt.n_component, max_length)
    else:
        y_mask_mat = torch.zeros(3 + 4 * opt.n_component, max_length)
    w_mask_mat = torch.zeros(30, max_length)
    c_mask_mat = torch.zeros(1, max_length)
    input_mat[0, :ins_length] = torch.Tensor(label[b'x_vals'])  # scale
    input_mat[1, :ins_length] = torch.Tensor(label[b'y_vals'])  # translation
    input_mat[2, :ins_length] = torch.Tensor(label[b'e_vals'])  # stop sign
    rot_mat[0, :ins_length] = torch.Tensor(label[b'r_vals'])  # rotation angle
    rot_mat[1, :ins_length] = torch.Tensor(label[b'rs_vals'])  # rotation axis
    if opt.loss_c is not None:
        cls_mat[0, :ins_length] = torch.Tensor(label[b'cls']) - 1   # class label 0 to opt.n_sem-1
    cls_mat = cls_mat.long()
    y_mask_mat[:, :ins_length] = 1.
    w_mask_mat[:, :ins_length] = 1.
    c_mask_mat[:, :ins_length] = 1.
    if ins_length == 0:
        input_mat[2, 0] = 1
    if opt.xyz == 3:
        # print('before', input_mat, rot_mat, cls_mat, c_mask_mat, max_length, ins_length)
        input_mat, rot_mat, cls_mat, c_mask_mat = xyz_one_step(input_mat, rot_mat, cls_mat, c_mask_mat,
                                                               max_length, ins_length)
        ins_length = ins_length // 3
        # print('after', input_mat, rot_mat, cls_mat, c_mask_mat, max_length, ins_length)
        if opt.out_r == 'class':
            rot_mat = rot_mat_to_theta_axis_class(rot_mat, test_init)
    if opt.test_loss == 'test_nn_loss':
        input_mat, rot_mat, cls_mat, c_mask_mat, prim_box_2d = replace_init(input_mat, rot_mat,
                                                                            cls_mat, c_mask_mat,
                                                                            init, test_init, ins_id,
                                                                            prim_box_2d)
    if opt.stage == 'ssign':
        input_mat = torch.ones(1) * ins_length // (3 // opt.xyz)
    if inverse:
        input_mat, rot_mat, cls_mat = get_inverse_mat(input_mat, rot_mat, cls_mat, ins_length)
    return input_mat, rot_mat, cls_mat, c_mask_mat, prim_box_2d


def rot_mat_to_theta_axis_class(rot_mat, test_init):
    # theta 0 - 26 (-0.78:0, 0:13, 0.78:26)
    # axis 0 - 3 (0: no rotation, 1: x, 2: y, 3: z)
    # print('before', rot_mat)
    assert opt.xyz == 3
    mean_r, std_r = test_init['mean_r'], test_init['std_r']
    length = rot_mat.size(1)
    rot_out = torch.zeros(2, length)
    for i in range(length):
        theta_3d = rot_mat[:3, i]
        axis_3d = rot_mat[3:, i]
        theta_3d = theta_3d * std_r[0, 0] + mean_r[0, 0]
        theta, axis = theta_axis_to_class(theta_3d, axis_3d, theta_dim=3)
        rot_out[:, i] = torch.Tensor([theta, axis])

        # if torch.sum(rot_mat[3:, i]) == 0:
        #     rot_out[0, i] = 13
        #     rot_out[1, i] = 0
        # else:
        #     axis = torch.argmax(rot_mat[3:, i])
        #     theta = rot_mat[axis.item(), i].item()
        #     theta = theta * std_r[0, 0] + mean_r[0, 0]
        #     theta = int(round((theta * 100))) // 6 + 13
        #     rot_out[0, i] = theta
        #     rot_out[1, i] = axis + 1
    rot_out = rot_out.long()
    # print('after', rot_out)
    return rot_out


def rot_gt_theta_axis_to_class(prim_set):
    theta_axis_class = []
    prim_set = torch.from_numpy(prim_set)
    for i in range(prim_set.size(0)):
        axis_3d = prim_set[i][16:19]
        theta = prim_set[i][19:]
        theta, axis = theta_axis_to_class(theta, axis_3d, theta_dim=1)
        theta_axis_class.append([theta, axis])
    return torch.Tensor(theta_axis_class).long()


def theta_axis_to_class(theta, axis_3d, theta_dim=3):
    if torch.sum(axis_3d) == 0:
        theta = 13
        axis = 0
    else:
        axis = torch.argmax(axis_3d).item()
        if theta_dim == 3:
            theta = theta[axis].item()
        else:
            theta = theta.item()
        theta = int(round((theta * 100))) // 6 + 13
        axis += 1
    return theta, axis


def xyz_one_step(input_mat, rot_mat, cls_mat, c_mask_mat, max_length, ins_length):
    length = max_length // 3
    input_mat_new = torch.zeros(2*opt.xyz+1, length)
    rot_mat_new = torch.zeros(2*opt.xyz, length)
    cls_mat_new = torch.zeros(1, length)
    c_mask_mat_new = torch.zeros(1, length)
    c_mask_mat_new[:, :ins_length//3] = 1.
    for i in range(0, max_length, 3):
        input_mat_new[0:3, i//3] = input_mat[0, i:i+3]
        input_mat_new[3:6, i//3] = input_mat[1, i:i+3]
        input_mat_new[6, i//3] = input_mat[2, i+2]  # 0 0 0 0 0 1
        rot_mat_new[0:3, i//3] = rot_mat[0, i:i+3]
        rot_mat_new[3:6, i//3] = rot_mat[1, i:i+3]
        cls_mat_new[0, i//3] = cls_mat[0, i]    # 0 0 0 1 1 1
    cls_mat_new = cls_mat_new.long()
    if ins_length == 0:
        input_mat_new[6, 0] = 1
    return input_mat_new, rot_mat_new, cls_mat_new, c_mask_mat_new


def replace_init(input_mat, rot_mat, cls_mat, c_mask_mat, init, test_init, ins_id, prim_box_2d=None):
    if opt.init_in is not None:
        prim_init = init
    else:
        prim_init = copy.deepcopy(test_init['test_sample'][:, ins_id])
    if 'SYNSet' in opt.check or 'NNCompute' in opt.check:
        print('prim_init', prim_init)
    # input_mat[:, 0] = torch.Tensor([prim_init[0], prim_init[1], 0])
    input_mat[:, 0] = torch.cat((torch.Tensor(prim_init[0:2 * opt.xyz]), torch.Tensor([0])), dim=0)
    if opt.init_source == 'given':
        rot_mat[:, 0] = torch.Tensor([prim_init[2], prim_init[3]])
    else:
        # rot_mat[:, 0] = torch.Tensor([prim_init[3], prim_init[4]])
        rot_mat[:, 0] = torch.Tensor(prim_init[2 * opt.xyz + 1:4 * opt.xyz + 1])
    cls_mat[:, 0] = torch.zeros(1).long()
    c_mask_mat[:, 0] = torch.ones(1).long()
    dim = 4*opt.xyz + 1
    if opt.loss_c is not None:
        dim += 1
        cls_mat[:, 0] = torch.Tensor([prim_init[dim - 1]]).long()
    if opt.box2d_source != 'oracle':
        for j in range(3//opt.xyz):
            # pass
            prim_box_2d[:, j] = torch.Tensor(prim_init[-4:])
    return input_mat, rot_mat, cls_mat, c_mask_mat, prim_box_2d


def get_init(labels):
    # deprecated
    # init = torch.zeros(6)
    # if opt.prim_norm is not None:
    #     raise NotImplementedError
    init = torch.zeros(10)
    if opt.init_in == 'mean':
        for i in range(len(labels)):
            init[0] += labels[i][b'x_vals'][0]
            init[1] += labels[i][b'y_vals'][0]
            init[3] += labels[i][b'r_vals'][0]
        init /= len(labels)
    init[5] = 1

    return init


def labels_add_init(labels, init):
    for i in range(len(labels)):
        labels[i][b'x_vals'].insert(0, init[0].item())
        labels[i][b'y_vals'].insert(0, init[1].item())
        labels[i][b'e_vals'].insert(0, init[2].item())
        labels[i][b'r_vals'].insert(0, init[3].item())
        labels[i][b'rs_vals'].insert(0, init[4].item())
        labels[i][b'cls'].insert(0, init[5].item())
    return labels


def get_max_length(labels, obj_class):
    if 'all' in obj_class:
        max_length = 0
        for cls in opt.file_names['obj_classes']:
            max_length_cls = get_max_length_one_class(labels[cls])
            if max_length < max_length_cls:
                max_length = max_length_cls
    else:
        max_length = get_max_length_one_class(labels)
    return max_length


def get_max_length_one_class(labels):
    max_length = 0
    for i in range(len(labels)):
        if max_length < len(labels[i][b'x_vals']):
            max_length = len(labels[i][b'x_vals'])
    return max_length


def normalize_prim(prim, mean_std_init):
    # prim numpy (n, 20)
    mean_x = mean_std_init['mean_x']
    mean_y = mean_std_init['mean_y']
    mean_r = mean_std_init['mean_r']
    std_x = mean_std_init['std_x']
    std_y = mean_std_init['std_y']
    std_r = mean_std_init['std_r']
    prim_out = []
    for i in range(prim.shape[0]):
        s_i = (prim[i][10:13] - mean_x[0, 0]) / std_x[0, 0]
        t_i = (prim[i][13:16] - mean_y[0, 0]) / std_y[0, 0]
        r_i = (prim[i][16:19] * prim[i][19] - mean_r[0, 0]) / std_r[0, 0]
        prim_i = np.concatenate((s_i, t_i, r_i), axis=0)
        prim_out.append(prim_i)
    return np.array(prim_out)


def get_sym(prim_r, voxel_scale):
    if prim_r[13] + prim_r[10] < voxel_scale / 2:
        sym = prim_r
        sym[13] = voxel_scale - prim_r[13] - prim_r[10]
        Rv = prim_r[16:19]
        theta = prim_r[19]
        Rv_t = np.argmax(np.abs(Rv))
        if Rv_t != 0:
            theta = -theta
        sym[19] = theta
    else:
        sym = np.zeros([20], dtype=int)
    return sym


def get_mean(shape, trans, scale, Rrot):
    #cen_mean = [14.343102,22.324961,23.012661]
    cen_mean = [trans[0]+shape[0]/2, trans[1]+shape[1]/2, trans[2]+shape[2]/2]
    # print(cen_mean)
    return cen_mean


def getVx(axis):
    vx = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    return vx


def get_rot_matrix(axis, theta):
    vx = getVx(axis)
    Rrot = math.cos(theta) * np.eye(3) + math.sin(theta) * vx + \
           (1 - math.cos(theta)) * np.array([axis], ).T * np.array([axis], )
    return Rrot


def prim_to_corners(prim_r):
    prim_pt_x = np.array([0, prim_r[10], prim_r[10], 0, 0, prim_r[10], prim_r[10], 0])
    prim_pt_y = np.array([0, 0, prim_r[11], prim_r[11], 0, 0, prim_r[11], prim_r[11]])
    prim_pt_z = np.array([0, 0, 0, 0, prim_r[12], prim_r[12], prim_r[12], prim_r[12]])
    prim_pt = [prim_pt_x, prim_pt_y, prim_pt_z]
    prim_pt = np.array(prim_pt)

    prim_pt = prim_pt.T + prim_r[13:16]
    prim_pt_mean = prim_pt.mean(axis=0)

    axis = prim_r[16:19]
    theta = prim_r[19]
    Rrot = get_rot_matrix(axis, theta)

    prim_pt -= prim_pt_mean
    prim_pt = prim_pt.dot(Rrot)
    prim_pt += prim_pt_mean
    if not opt.global_denorm:
        prim_pt[np.where(prim_pt > opt.v_size)] = opt.v_size
        prim_pt = prim_pt / opt.v_size
    return prim_pt


def obb_to_aabb(prim_pt):
    min_xyz = np.min(prim_pt, axis=0)
    max_xyz = np.max(prim_pt, axis=0)
    prim_pt_x = np.array([min_xyz[0], max_xyz[0], max_xyz[0], min_xyz[0], min_xyz[0], max_xyz[0], max_xyz[0], min_xyz[0]])
    prim_pt_y = np.array([min_xyz[1], min_xyz[1], max_xyz[1], max_xyz[1], min_xyz[1], min_xyz[1], max_xyz[1], max_xyz[1]])
    prim_pt_z = np.array([min_xyz[2], min_xyz[2], min_xyz[2], min_xyz[2], max_xyz[2], max_xyz[2], max_xyz[2], max_xyz[2]])
    prim_pt = [prim_pt_x, prim_pt_y, prim_pt_z]
    prim_pt = np.array(prim_pt).T
    return prim_pt


def prim_all_to_cornerset(prim_all):
    cornerset = []
    for j in range(prim_all.shape[0]):
        prim_r = prim_all[j]
        prim_pt = prim_to_corners(prim_r)
        if 'aabb' in opt.metric:
            prim_pt = obb_to_aabb(prim_pt)
        prim_pt
        cornerset.append(prim_pt)
    return cornerset


def remove_empty_prim_in_primset_sym(prim, cls=None):
    new_prim = []
    new_cls = []
    for n in range(prim.shape[0]):
        if np.sum(prim[n, :]):
            new_prim.append(prim[n, :])
            if cls is not None:
                new_cls.append(cls[n])
    new_prim = np.array(new_prim)
    new_cls = np.array(new_cls)
    return new_prim, new_cls


def res_obj_to_gt_prims(i, res):
    # res = copy.deepcopy(res)
    res_x = res['x'][i * 4:i * 4 + 4, :]  # 269*4,100
    res_cls = res['cls'][i]  # 269, 100
    res_box2d = res['box2d'][i * 4:i * 4 + 4, :]  # 269*4, 100
    res_rot_prob = res['rot_prob'][i * 31:i * 31 + 31, :]  # 269*31, 100
    res_cls_prob = res['cls_prob'][i * opt.n_sem:i * opt.n_sem + opt.n_sem, :]  # 269*6, 100
    voxel_scale = opt.v_size
    prim_all = []
    cls_all = []
    box2d_all = []
    rot_prob_all = []
    cls_prob_all = []
    sym_tag = []
    res_prim = res_x[0:2, :]
    res_rot = res_x[2, :]
    # res_sym = res[3, :]
    stop_idx = [ind for (ind, x) in enumerate(res_prim[0, :]) if x == 0][0]
    num_kill = 0
    for col in range(0, stop_idx - 2, 3):
        prim_rot = res_rot[col:col + 3]
        prim_r = np.concatenate((res_prim[0, col:col + 3], res_prim[1, col:col + 3], prim_rot), axis=0)
        # sym_r = res_sym[col:col + 3]
        euler = prim_r[6:9]
        # Rv_y = np.max(np.abs(Rv))
        axis, theta = rot_euler_to_axis_theta(euler)
        prim_r = np.concatenate((np.zeros([10]), prim_r[0:6], axis, [theta]), axis=0)
        if res_cls[col] == opt.n_sem:   ## background class
            if len(prim_all) == 0 and col + 3 >= stop_idx - 2:
                res_cls[col] = 0
            else:
                num_kill += 1
                continue
        prim_all.append(prim_r)
        cls_all.append(res_cls[col])
        box2d_all.append(res_box2d[:, col])
        rot_prob_all.append(res_rot_prob[:, col])
        cls_prob_all.append(res_cls_prob[:, col])
        sym_tag.append(0)
        if prim_r[13] + prim_r[10] >= voxel_scale / 2 or opt.global_denorm or opt.full_model or opt.pqnet:
            continue
        else:
            assert not opt.full_model
            prim_s = prim_r.copy()
            prim_s = get_sym(prim_s, voxel_scale)
            prim_all.append(prim_s)
            cls_all.append(res_cls[col])
            rot_prob_all.append(res_rot_prob[:, col])
            cls_prob_all.append(res_cls_prob[:, col])
            sym_tag.append(1)
    # print('pred len', len(list(range(0, stop_idx - 2, 3))), 'num kill', num_kill, 'remain', len(prim_all))
    return (np.array(prim_all), np.array(cls_all) + 1, np.array(box2d_all),
            np.array(rot_prob_all), np.array(cls_prob_all), np.array(sym_tag))


def rot_euler_to_axis_theta(euler):
    max_id = np.argmax(np.abs(euler))
    theta = euler[max_id]
    axis = np.zeros([3], dtype=int)
    axis[max_id] = 1
    return axis, theta


def visualize_box2d_pred(prim_box_2d, id_img_ori, image, transform):
    prim_box_2d = np.concatenate((prim_box_2d[:, 0:2], prim_box_2d[:, 2:4]), axis=0)  # box to kp
    sample = {'image': image, 'label': copy.deepcopy(prim_box_2d)}
    if transform is not None:
        sample = transform(sample)
    visualize_img(sample['image'], torch.from_numpy(prim_box_2d), id_img_ori, transformed=True)
    # sample = transform_image_label(id_img_ori, image, label=prim_box_2d, transform=transform)


def count_rotation_gt(data_dir):
    import pdb
    pdb.set_trace()
    rotation = {}
    gt = scipy.io.loadmat(os.path.join(data_dir, opt.file_names['primset']))
    gt = gt['primset']  # 216,1 struct
    num = gt.shape[0]
    for i in range(num):
        gt_prim_ori = gt[i, 0]['ori'][0, 0]
        gt_prim_sym = gt[i, 0]['sym'][0, 0]
        gt_cls_ori = gt[i, 0]['cls'][0, 0][0]
        if np.sum(gt_prim_sym):
            gt_prim_sym, gt_cls_sym = remove_empty_prim_in_primset_sym(gt_prim_sym, gt_cls_ori)
            gt_prim = np.vstack((gt_prim_ori, gt_prim_sym))
            gt_cls = np.hstack((gt_cls_ori, gt_cls_sym))
        else:
            gt_prim = gt_prim_ori
            gt_cls = gt_cls_ori
        for j in range(gt_prim.shape[0]):
            rot = gt_prim[j, -1]
            rot = round(rot * 100) / 100
            import math
            if abs(rot) > math.pi/4:    #gt_cls[j] == 6:
                print(i+1, rot, j+1, gt_cls[j])
            if rot not in rotation.keys():
                rotation[rot] = 1
            else:
                rotation[rot] += 1
    for rot in sorted(rotation.keys()):
        print('theta:', rot, '     number of prims:', rotation[rot])
    import pdb;pdb.set_trace()


def get_g1_in_size(node):
    g1_in_size = 0
    if 'img' in node:
        g1_in_size += get_img_feature_size(node)
    if 'h' in node:
        g1_in_size += opt.hid_size * opt.hid_layer
    if 'n' in node:
        g1_in_size += opt.f_dim
    if 'st' in node:
        g1_in_size += 2 * opt.xyz
    if 'r' in node:
        g1_in_size += 1 * opt.xyz
    if 'cls' in node:
        g1_in_size += opt.n_sem + int(opt.len_adjust)
    if 'cot' in node:
        g1_in_size += opt.n_sem + int(opt.len_adjust)
    return g1_in_size


def get_img_feature_size(node):
    if opt.model_name in ['resnet18', 'resnet34']:
        expansion = 1
    elif opt.model_name in ['resnet50', 'resnet101', 'resnet152']:
        expansion = 4
    else:
        raise NotImplementedError
    if 'img_f' in node:
        return 512 * expansion
    if 'img_global' in node:
        return opt.con_size
    c_size = opt.con_size
    if opt.box2d_pos == '0' and opt.box2d_size > 0:
        c_size += opt.box2d_size
        if opt.box2d_en == 'hard':
            c_size += 4
    if opt.bbox_con[4:] in ['all', 'bbox']:
        c_size += opt.n_sem * opt.n_para
    if opt.bbox_con[4:] in ['all', 'exist']:
        c_size += opt.n_sem
    return c_size


def get_sequence_len_all(input_mat):
    batch_size = input_mat.size(0)
    len_all = [0 for _ in range(batch_size)]
    input_max_len = input_mat.size(2)
    for b in range(batch_size):
        for i in range(input_max_len, 0, -1):
            if torch.sum(torch.abs(input_mat[b, :, i - 1])) != 0:
                len_all[b] = i
                break
    assert len(len_all) == batch_size
    return len_all


def get_matched_cls_gt(outputs, targets, c_mask_mat, length):
    st_criterion = nn.L1Loss().cuda()
    pred, gt = [], []
    for i in range(len(outputs)):
        pred.append(outputs[i][0])
    for i in range(len(targets)):
        gt.append(targets[i][0])
    batch_size = pred[0].size(0)
    cls_gt_all, res_gt_id_all = [], []
    for b in range(batch_size):
        gt_length = torch.sum(c_mask_mat[b]).int().item()
        if opt.len_source != 'None':
            res_length = length[b, 0].int().item()
        else:
            res_length = gt_length
        if opt.reg_init == 'None':
            gt_length -= 1
            res_length -= 1
        # print(gt_length, len(targets), res_length, len(outputs))
        assert gt_length <= len(targets) and res_length <= len(outputs)
        cls_gt = [0 for _ in range(res_length)]
        res_gt_id_out = [0 for _ in range(res_length)]
        res_id = list(range(res_length))
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
            for i in range(res_length):
                dis_min_id = np.argmin(dis)
                res_i_i = dis_min_id // dis.shape[1]
                gt_i_i = dis_min_id % dis.shape[1]
                res_i = res_id.pop(res_i_i)
                if not opt.one2many:# and opt.len_adjust: # every node should predict sth, inconsistent with loss
                    gt_i = gt_id.pop(gt_i_i)
                else:
                    gt_i = gt_id[gt_i_i]
                # cls_gt.append(targets[gt_i][2][b:b+1])
                cls_gt[res_i] = targets[gt_i][2][b:b+1]
                res_gt_id_out[res_i] = gt_i
                dis = np.delete(dis, res_i_i, axis=0)
                if not opt.one2many:# and opt.len_adjust:
                    dis = np.delete(dis, gt_i_i, axis=1)
                if dis.shape[0] == 0 or dis.shape[1] == 0:
                    break
        if opt.len_adjust:  # inexistent part loss
            for res_i in res_id:
                # cls_gt.append(torch.Tensor([opt.n_sem]).long().cuda())
                cls_gt[res_i] = torch.Tensor([opt.n_sem]).long().cuda()
                res_gt_id_out[res_i] = -1
        else:
            for res_i in res_id:    # no loss should be computed later, if matching results are the same, not guaranteed
                # cls_gt.append(torch.Tensor([opt.n_sem]).long().cuda())
                cls_gt[res_i] = torch.Tensor([0]).long().cuda()
                res_gt_id_out[res_i] = -1
        cls_gt_all.append(cls_gt)
        res_gt_id_all.append(res_gt_id_out)
    return cls_gt_all, res_gt_id_all


def combine_bi_direction(proposal, proposal_inv, prev_c_flag=False):
    assert proposal['length'] == proposal_inv['length']
    assert len(proposal['features']) == len(proposal_inv['features'])
    assert len(proposal['outputs']) == len(proposal_inv['outputs'])
    assert len(proposal['prev_h'].keys()) == len(proposal_inv['prev_h'].keys())
    proposal['length'] += proposal_inv['length']
    proposal['features'] += proposal_inv['features']
    proposal['outputs'] += proposal_inv['outputs']
    length = len(proposal['prev_h'].keys())
    for key, value in proposal_inv['prev_h'].items():
        if key != 0:
            proposal['prev_h'][key + length - 1] = proposal_inv['prev_h'][key]
    if prev_c_flag:
        assert len(proposal['prev_c'].keys()) == len(proposal_inv['prev_c'].keys())
        for key, value in proposal_inv['prev_c'].items():
            if key != 0:
                proposal['prev_c'][key + length - 1] = proposal_inv['prev_c'][key]
    # pdb.set_trace()
    return proposal




def res_obj_to_gt_prims_lstm2(i, res):
    # res = copy.deepcopy(res)
    res_x = res['x'][i * 4:i * 4 + 4, :]  # 269*4,100
    res_cls = res['cls'][i:i+1]  # 269, 100
    res_box2d = res['box2d'][i * 4:i * 4 + 4, :]  # 269*4, 100
    res_rot_prob = res['rot_prob'][i * 31:i * 31 + 31, :]  # 269*31, 100
    res_cls_prob = res['cls_prob'][i * opt.n_sem:i * opt.n_sem + opt.n_sem, :]  # 269*6, 100
    voxel_scale = opt.v_size
    prim_all = []
    cls_all = []
    box2d_all = []
    rot_prob_all = []
    cls_prob_all = []
    sym_tag = []
    res_prim = res_x[0:2, :]
    res_rot = res_x[2, :]
    # res_sym = res[3, :]
    stop_idx = [ind for (ind, x) in enumerate(res_prim[0, :]) if x == 0][0]
    num_kill = 0
    for col in range(0, stop_idx - 2, 3):
        # prim_rot = res_rot[col:col + 3]
        prim_r = np.concatenate((res_prim[0, col:col + 3], res_prim[1, col:col + 3]), axis=0)
        prim_all.append([torch.from_numpy(copy.deepcopy(prim_r)), copy.deepcopy(res_cls_prob[:, col])])
    return prim_all, (torch.from_numpy(res_x), torch.from_numpy(res_cls), 
        torch.from_numpy(res_box2d), torch.from_numpy(res_rot_prob), 
        torch.from_numpy(res_cls_prob))


def drop_outputs_lstm2(outputs, targets, length):
    outputs_a, outputs_b = [], []
    dis_paired = []
    st_criterion = nn.L1Loss().cuda()
    pred, gt = [], []
    for i in range(len(outputs)):
        pred.append(outputs[i][0])
    for i in range(len(targets)):
        gt.append(targets[i][0])
    # batch_size = pred[0].size(0)
    batch_size = 1
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
                    dis[i, j] = st_criterion(pred[i], gt[j]).item()
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
            res_cls = torch.from_numpy(outputs[res_i][1])#outputs[res_i][2][b:b + 1]
            gt_cls = torch.from_numpy(targets[gt_i][1])#targets[gt_i][2][b:b + 1]
            # pdb.set_trace()
            if opt.drop_test == 1.5:
                pdb.set_trace()
                if random.random() < 0.5:# or torch.argmax(gt_cls).item() == opt.n_sem:
                    outputs_a.append(outputs[res_i])
                    outputs_b.append(targets[gt_i])
                else:
                    outputs_a.append(targets[gt_i])
                    outputs_b.append(outputs[res_i])
            else:
                if torch.max(res_cls) > torch.max(gt_cls):# or torch.argmax(gt_cls).item() == opt.n_sem:
                    outputs_a.append(res_i)
                    outputs_b.append(gt_i)
                else:
                    outputs_a.append(gt_i)
                    outputs_b.append(res_i)
            dis_paired.append(dis_min)
            if dis.shape[0] == 0 or dis.shape[1] == 0:
                break
    return outputs_a


def lstm2_final():
    #root = '/root/final_prim/lstm2'
    root = '/root/model_split/primitive-based_3d/3dprnn_pytorch/expprnn'
    save_dir = os.path.join(root, 'p02c01_0_v1_405')
    fdir = os.path.join(root, 'p02c01_0_v1_400', 'test_res_mn_{}.mat'.format(opt.obj_class))
    bdir = os.path.join(root, 'p02c01_01_v1_400', 'test_res_mn_{}.mat'.format(opt.obj_class))

    # save_dir = os.path.join(root, 'dag02c02_2_v1_405')
    # fdir = os.path.join(root, 'dag02c01_22_v1_400', 'test_res_mn_{}.mat'.format(opt.obj_class))
    # bdir = os.path.join(root, 'dag02c01_33_v1_400', 'test_res_mn_{}.mat'.format(opt.obj_class))
    
    # save_dir = os.path.join(root, 'p02c02_1_v1_405')
    # fdir = os.path.join(root, 'p02c01_2_v1_400', 'test_res_mn_{}.mat'.format(opt.obj_class))
    # bdir = os.path.join(root, 'p02c01_3_v1_400', 'test_res_mn_{}.mat'.format(opt.obj_class))
    ffile = scipy.io.loadmat(fdir)
    bfile = scipy.io.loadmat(bdir)
    
    num = ffile['x'].shape[0] // 4
    print('num of objects', num)
    # assert num == 810
    n_sem = opt.n_sem
    rs_res = torch.zeros(4 * num, opt.max_len)
    cls_res = - torch.ones(num, opt.max_len).long()
    invalid_res = - torch.ones(num, opt.max_len).long()
    box2d_res = torch.zeros(4 * num, opt.max_len)
    cls_prob_res = torch.zeros(n_sem * num, opt.max_len)
    rot_prob_res = torch.zeros(31 * num, opt.max_len)

    for i_batch in range(num):
        fprim, fall = res_obj_to_gt_prims_lstm2(i_batch, ffile)
        bprim, ball = res_obj_to_gt_prims_lstm2(i_batch, bfile)
        outputs = fprim + bprim
        assert len(fprim) == len(bprim)
        length = len(outputs)
        out = drop_outputs_lstm2(copy.deepcopy(outputs), copy.deepcopy(outputs), length)

        rs = torch.zeros(4, opt.max_len)
        cls = - torch.ones(1, opt.max_len).long()
        box2d = torch.zeros(4, opt.max_len)
        cls_prob = torch.zeros(n_sem, opt.max_len)
        rot_prob = torch.zeros(31, opt.max_len)

        count = 0
        for ii in out:
            if ii < len(fprim):
                rs[:, count*3:count*3+3] = fall[0][:, ii*3:ii*3+3]
                cls[:, count*3:count*3+3] = fall[1][:, ii*3:ii*3+3]
                box2d[:, count*3:count*3+3] = fall[2][:, ii*3:ii*3+3]
                rot_prob[:, count*3:count*3+3] = fall[3][:, ii*3:ii*3+3]
                cls_prob[:, count*3:count*3+3] = fall[4][:, ii*3:ii*3+3]
            else:
                ii = ii - len(fprim)
                rs[:, count*3:count*3+3] = ball[0][:, ii*3:ii*3+3]
                cls[:, count*3:count*3+3] = ball[1][:, ii*3:ii*3+3]
                box2d[:, count*3:count*3+3] = ball[2][:, ii*3:ii*3+3]
                rot_prob[:, count*3:count*3+3] = ball[3][:, ii*3:ii*3+3]
                cls_prob[:, count*3:count*3+3] = ball[4][:, ii*3:ii*3+3]
            count += 1

        rs_res[i_batch * 4: i_batch * 4 + 4, :] = rs
        cls_res[i_batch, :] = cls
        box2d_res[i_batch * 4: i_batch * 4 + 4, :] = box2d
        cls_prob_res[i_batch * n_sem: i_batch * n_sem + n_sem, :] = cls_prob
        rot_prob_res[i_batch * 31: i_batch * 31 + 31, :] = rot_prob
    res = {'x': rs_res.numpy(), 'cls': cls_res.numpy(), 
           'box2d': box2d_res.numpy(), 'cls_prob': cls_prob_res.numpy(),
           'rot_prob': rot_prob_res.numpy()}

    scipy.io.savemat(save_dir + '/test_res_mn_{}.mat'.format(opt.obj_class), res)

