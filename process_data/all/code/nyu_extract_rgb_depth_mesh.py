import os
import numpy as np
import scipy.io
import cv2
import pdb


def save_mesh_obj_from_mat():
    root = os.path.abspath('.')
    in_dir = os.path.join(root, '../input/nyu/model')
    save_dir = os.path.join(root, '../input/nyu/obj')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for tmp_path, dirs, files in os.walk(in_dir):
        if len(dirs) == 0:
            files = sorted(files)
            for file in files:
                if file[-4:] != '.mat':
                    continue
                print(file)
                file_dir = os.path.join(in_dir, file)
                obj_file_name = os.path.join(save_dir, file.replace('mat', 'obj'))
                write_obj_one(file_dir, obj_file_name)


def write_obj_one(file_dir, obj_file_name, scale_xyz=[1, 1, 1], cls_obj=None, mesh_mat=None):
    if cls_obj != 'night stand':
        mesh_mat = scipy.io.loadmat(file_dir)['comp']
    mesh_obj = {'v': [], 'f': []}
    v_id = 0
    for i in range(mesh_mat.shape[1]):
        part = mesh_mat[0, i]
        for v_i in range(part['vertices'][0, 0].shape[0]):  # n*3
            mesh_obj['v'].append(part['vertices'][0, 0][v_i])
        for f_i in range(part['faces'][0, 0].shape[0]):  # m*3
            mesh_obj['f'].append(part['faces'][0, 0][f_i].astype(int) + v_id)
        v_id += part['vertices'][0, 0].shape[0]
        # if v_id > 255 and file == 'bed_45.mat':
        #     pdb.set_trace()
    obj_file = open(obj_file_name, 'w')
    mesh_obj = {'v': np.array(mesh_obj['v']), 'f': np.array(mesh_obj['f'])}
    for i in range(3):
        mesh_obj['v'][:, i] *= scale_xyz[i]
    if cls_obj == 'night stand':
        mean_xyz = np.mean(mesh_obj['v'], axis=0)
        len_xyz = np.max(mesh_obj['v'], axis=0) - np.min(mesh_obj['v'], axis=0)
        mesh_obj['v'] -= mean_xyz
        mesh_obj['v'] /= np.max(len_xyz)
        mesh_obj['v'] += 0.5

    if max(abs(np.max(mesh_obj['v'])), abs(np.min(mesh_obj['v']))) > 0.5:
        print('>' * 30)
    # if max(abs(np.max(mesh_obj['v'])), abs(np.min(mesh_obj['v']))) < 0.48:
    #     print('<' * 30)
    # print(np.max(mesh_obj['v'][:, 0]), np.max(mesh_obj['v'][:, 1]), np.max(mesh_obj['v'][:, 2]))
    # print(np.min(mesh_obj['v'][:, 0]), np.min(mesh_obj['v'][:, 1]), np.min(mesh_obj['v'][:, 2]))
    for v_i in range(mesh_obj['v'].shape[0]):
        vv = mesh_obj['v'][v_i, :].astype(float)
        obj_file.write('v ' + str(vv[0]) + ' ' + str(vv[1]) + ' ' + str(vv[2]) + '\n')
    for f_i in range(mesh_obj['f'].shape[0]):
        ff = mesh_obj['f'][f_i, :]
        obj_file.write('f ' + str(ff[0]) + ' ' + str(ff[1]) + ' ' + str(ff[2]) + '\n')
    obj_file.close()


def binary_mask_to_bbox(mask):
    xs = np.sum(mask, axis=0)
    ys = np.sum(mask, axis=1)
    xs = np.nonzero(xs)
    ys = np.nonzero(ys)
    x_min, x_max = np.min(xs), np.max(xs)
    y_min, y_max = np.min(ys), np.max(ys)
    return x_min, y_min, x_max, y_max


def crop_bbox(image, bbox):
    thresh = 0.1
    h, w = image.shape[0], image.shape[1]
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


def check_crop(image, depths, mask_i, image_i, depths_i, visual_dir, file_name):
    import matplotlib.pyplot as plt
    I = plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title('image')
    plt.subplot(2, 3, 2)
    plt.imshow(depths)
    plt.title('depths')
    plt.subplot(2, 3, 3)
    plt.imshow(mask_i)
    plt.title('mask_i')
    plt.subplot(2, 3, 4)
    plt.imshow(image_i)
    plt.title('image_i')
    plt.subplot(2, 3, 5)
    plt.imshow(depths_i)
    plt.title('depths_i')
    save_dir = os.path.join(visual_dir, file_name)
    plt.savefig(save_dir)
    plt.close('all')


def extract_data(cls):
    root = os.path.abspath('.')
    in_dir = os.path.join(root, '../input/nyu')
    out_dir = os.path.join(root, '../output', cls)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    visual_dir = os.path.join(root, '../visual/nyu', cls)
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)
    scene_splits_dir = os.path.join(in_dir, 'splits.mat')
    scene_splits = scipy.io.loadmat(scene_splits_dir)
    scene_splits = {'train': scene_splits['trainNdxs'][:, 0].tolist(),
                    'test': scene_splits['testNdxs'][:, 0].tolist()}
    obj_count = 0

    img_idxs = []
    voxel_idxs = []
    voxel_txt_dir = os.path.join(out_dir, 'voxeltxt')
    if not os.path.exists(voxel_txt_dir):
        os.makedirs(voxel_txt_dir)
    txt_path = {x: os.path.join(voxel_txt_dir, '{}.txt'.format(x))
               for x in ['train', 'test']}
    f = {x: open(txt_path[x], 'w') for x in ['train', 'test']}
    f_obj = open(os.path.join(out_dir, 'voxels_dir_{}.txt'.format(cls)), 'w')
    f_scene = open(os.path.join(out_dir, 'scenes_dir_{}.txt'.format(cls)), 'w')

    depth_dir, img_dir = None, None
    if '3dprnn' in cls:
        suffix = '.mat'
        depth_dir = os.path.join(out_dir, 'images_crop_object')
        if not os.path.exists(depth_dir):
            os.makedirs(depth_dir)
    else:
        suffix = '.png'
        img_dir = os.path.join(out_dir, 'images_crop_object')
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

    obj_dir = os.path.join(out_dir, 'obj')
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)

    data_dir = os.path.join(in_dir, 'rgb_depth_mesh')
    for tmp_path, dirs, files in os.walk(data_dir):
        if len(dirs) == 0:
            files = sorted(files)
            for file in files:
                if file[-4:] != '.mat':
                    continue
                print(obj_count, file)
                file_id = int(file.split('_')[0])
                file_dir = os.path.join(data_dir, file)
                scene_mat = scipy.io.loadmat(file_dir)['model'][0, 0]

                image = scene_mat['data'][0, 0]['image']
                depths = scene_mat['data'][0, 0]['depths']
                masks = scene_mat['data'][0, 0]['masks']

                objects = scene_mat['objects']
                for o_i in range(objects.shape[1]):
                    object_i = objects[0, o_i]
                    uid_i = object_i['uid']
                    model_i = object_i['model'][0, 0]
                    if model_i['type'][0, 0][0] == 'furniture' or model_i['label'][0, 0][0] == 'night stand':
                        if model_i['label'][0, 0][0] == 'night stand':
                            if 'night_stand' not in cls:
                                continue
                            mesh_mat_file_i = file
                        else:
                            if model_i['basemodel'][0, 0][0].split('_')[0] not in cls:
                                continue
                            mesh_mat_file_i = model_i['basemodel'][0, 0][0]
                            scale_xyz_i = model_i['scale_xyz'][0, 0][0]

                        f_obj.write(mesh_mat_file_i + '\n')
                        f_scene.write(file + '  ' + str(o_i) + '  ' + '\n')
                        mask_i = (masks == uid_i)
                        bbox_i = binary_mask_to_bbox(mask_i)
                        mask_i = crop_bbox(mask_i, bbox_i)
                        image_i = crop_bbox(image, bbox_i)
                        image_i = (image_i * 255).astype(np.uint8)
                        depths_i = crop_bbox(depths, bbox_i)
                        obj_count += 1

                        obj_id_i = ('0000' + str(obj_count))[-4:]
                        img_idxs.append(obj_count)
                        voxel_idxs.append(obj_count)
                        if file_id in scene_splits['train']:
                            f['train'].write(obj_id_i + suffix +'\n')
                        elif file_id in scene_splits['test']:
                            f['test'].write(obj_id_i + suffix + '\n')
                        else:
                            raise NotImplementedError
                        depths_i = depths_i * mask_i
                        check_crop(image, depths, mask_i, image_i, depths_i,
                                   visual_dir, obj_id_i + '.png')

                        if '3dprnn' in cls:
                            image_dir_i = os.path.join(depth_dir, obj_id_i + suffix)
                            scipy.io.savemat(image_dir_i, {'depth': depths_i})
                        else:
                            image_dir_i = os.path.join(img_dir, obj_id_i + suffix)
                            cv2.imwrite(image_dir_i, cv2.cvtColor(image_i, cv2.COLOR_RGB2BGR))

                        mesh_mat_dir_i = os.path.join(in_dir, 'model', mesh_mat_file_i)
                        obj_dir_i = os.path.join(obj_dir, obj_id_i + '.obj')
                        if model_i['label'][0, 0][0] == 'night stand':
                            write_obj_one(None, obj_dir_i, cls_obj='night stand', mesh_mat=object_i['mesh'][0, 0]['comp'][0, 0])
                        else:
                            write_obj_one(mesh_mat_dir_i, obj_dir_i, scale_xyz=scale_xyz_i)
                        scipy.io.savemat(os.path.join(out_dir, 'img_voxel_idxs.mat'),
                                         {'img_idxs': np.array(img_idxs), 'voxel_idxs': np.array(voxel_idxs)})
                # if obj_count > 40:
                #     break

    f['train'].close()
    f['test'].close()
    f_obj.close()
    f_scene.close()



if __name__ == '__main__':
    # save_mesh_obj_from_mat()
    cls = '3dprnnnyunight_stand'
    extract_data(cls)
