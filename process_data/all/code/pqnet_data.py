import h5py
import numpy as np
import os
import json
import scipy.io
import ipdb
import json


def generate_original_part_voxel_pq(category):
    root = '/root/model_split/PQ-NET'
    # category = 'Chair'
    vox_dim = 64
    pqroot = os.path.join(root, 'data')

    shape_names = sorted(os.listdir(os.path.join(pqroot, category)))
    shape_names = [x for x in shape_names if x[-3:] == '.h5']
    original_paths = [os.path.join(pqroot, 'old', category, name) for name in shape_names]
    voxel_paths = [os.path.join(pqroot, 'original_parts_voxel', category, name) for name in shape_names]

    for i in range(len(original_paths)):
        name = shape_names[i]
        opath_i = original_paths[i]
        vpath_i = voxel_paths[i]
        with h5py.File(opath_i, 'r') as fo:
            parts_voxel_scaled = fo['parts_voxel_scaled{}'.format(vox_dim)][:]
            n_parts = fo.attrs['n_parts']
            with h5py.File(vpath_i, 'a') as fv:
                # fo.create_dataset('shape_voxel{}'.format(vox_dim), shape=(vox_dim, vox_dim, vox_dim),
                #     dtype=np.uint8, data=shape_voxel, compression=9)
                fv.create_dataset('parts_voxel{}'.format(vox_dim), shape=(n_parts, vox_dim, vox_dim, vox_dim),
                    dtype=np.bool, data=parts_voxel_scaled, compression=9)
                # fp.create_dataset('parts_voxel_scaled{}'.format(vox_dim), shape=(n_parts, vox_dim, vox_dim, vox_dim),
                #                   dtype=np.bool, data=parts_voxel_scaled, compression=9)
                fv.attrs['n_parts'] = n_parts
                # fv.attrs['name'] = name.encode('utf-8')


def check_original_part_to_box(category):
    root = '/root/model_split/PQ-NET'
    # category = 'Lamp'
    vox_dim = 64
    pqroot = os.path.join(root, 'data')

    shape_names = sorted(os.listdir(os.path.join(pqroot, category)))
    shape_names = [x for x in shape_names if x[-3:] == '.h5']
    # original_paths = [os.path.join(pqroot, 'old', category, name) for name in shape_names]
    # voxel_paths = [os.path.join(pqroot, 'original_parts_voxel', category, name) for name in shape_names]
    original_paths = [os.path.join(pqroot, category, name) for name in shape_names]
    voxel_paths = original_paths

    for i in range(len(original_paths)):
        name = shape_names[i]
        opath_i = original_paths[i]
        vpath_i = voxel_paths[i]
        # opath_i = '/root/model_split/PQ-NET/data/old/Chair/179.h5'
        # vpath_i = opath_i
        with h5py.File(opath_i, 'r') as fo:
            with h5py.File(vpath_i, 'r') as fv:
                print(fo['parts_voxel_scaled{}'.format(vox_dim)][:].sum(),
                    fv['parts_voxel_scaled{}'.format(vox_dim)][:].sum(),
                    (fo['parts_voxel_scaled{}'.format(vox_dim)][:] != fv['parts_voxel_scaled{}'.format(vox_dim)][:]).sum())
                print(fo['size'][:])
                print(fv['size'][:])
                print(fo['scales'][:])
                print(fv['scales'][:])
                print(fo['translations'][:])
                print(fv['translations'][:])
                
                # shape_voxel = fo['shape_voxel64'][:]
                shape_voxel = fo['shape_voxel'][:]
                for ii in range(len(np.unique(shape_voxel))):
                    nonzeros = np.nonzero((shape_voxel == ii).astype(int))
                    x0, x1 = nonzeros[0].min(), nonzeros[0].max() + 1
                    y0, y1 = nonzeros[1].min(), nonzeros[1].max() + 1
                    z0, z1 = nonzeros[2].min(), nonzeros[2].max() + 1
                    size_ori = [x1 - x0, y1 - y0, z1 - z0]
                    trans_ori = [(x0 + x1) / 2, (y0 + y1) / 2,  (z0 + z1) / 2]
                    print(size_ori)
                    print(trans_ori)

                if (fo['translations'][:] + fo['size'][:] / 2 > 64).any():
                    print(opath_i)
                    print(fo['translations'][:] + fo['size'][:] / 2)

                ipdb.set_trace()
                tmp = 1

                # print((fo['parts_voxel_scaled{}'.format(vox_dim)][:] != fv['parts_voxel{}'.format(vox_dim)][:]).sum())
                # (fo['scales'][:] != fv['scales'][:]).sum()
                # (fo['size'][:] != fv['size'][:]).sum()
                # (fo['translations'][:] != fv['translations'][:]).sum()
                # (fo.attrs['n_parts'] != fv.attrs['n_parts']).sum()


def save_mat_for_visualization(category):
    root = '/root/model_split/PQ-NET'
    # category = 'Lamp'
    vox_dim = 64
    pqroot = os.path.join(root, 'data')

    shape_names = sorted(os.listdir(os.path.join(pqroot, category)))
    shape_names = [x for x in shape_names if x[-3:] == '.h5']
    original_paths = [os.path.join(pqroot, 'old', category, name) for name in shape_names]
    voxel_paths = [os.path.join(pqroot, 'original_parts_voxel', category, name) for name in shape_names]

    for i in range(len(original_paths)):
        name = shape_names[i]
        opath_i = original_paths[i]
        vpath_i = voxel_paths[i]
        spath_i = vpath_i[:-3] + '.mat'
        with h5py.File(opath_i, 'r') as fo:
            with h5py.File(vpath_i, 'r') as fv:
                obj_vox = fo['shape_voxel{}'.format(vox_dim)][:]
                part_vox_o = fo['parts_voxel_scaled{}'.format(vox_dim)][:]
                part_vox_v = fv['parts_voxel_scaled{}'.format(vox_dim)][:]
        scipy.io.savemat(spath_i, {'obj_vox': obj_vox, 'part_vox_o': part_vox_o, 'part_vox_v': part_vox_v})


def save_json_and_mat(category, save_mat=False):
    # root = '/root/model_split/PQ-NET'
    root = '/root/model_split/primitive-based_3d/pqnet/PQ-NET'
    vox_dim = 64
    pqroot = os.path.join(root, 'data')
    check_root = os.path.join(pqroot, 'check', category)
    if not os.path.exists(check_root):
        os.makedirs(check_root)
    class_info = {}
    max_n_parts = 0

    shape_names = sorted(os.listdir(os.path.join(pqroot, category)))
    shape_names = [x for x in shape_names if x[-3:] == '.h5']
    original_paths = [os.path.join(pqroot, category, name) for name in shape_names]
    check_paths = [os.path.join(check_root, name[:-3] + '.mat') for name in shape_names]

    for i in range(len(original_paths)):
        name = shape_names[i]
        opath_i = original_paths[i]
        cpath_i = check_paths[i]
        with h5py.File(opath_i, 'r') as fo:
            # if i==38:
            #     import ipdb; ipdb.set_trace()
            #     print(i, fo['points_16'][:].shape, fo['points_32'][:].shape, fo['points_64'][:].shape)
            n_parts = int(fo.attrs['n_parts'])
            if n_parts > max_n_parts:
                max_n_parts = n_parts
            class_info[name.split('.')[0]] = n_parts
            shape_voxel = fo['shape_voxel'][:]
            parts_voxel64 = fo['parts_voxel64'][:]
            parts_voxel_scaled64 = fo['parts_voxel_scaled64'][:]
        if save_mat:
            scipy.io.savemat(cpath_i, {'obj_vox': shape_voxel, 'part_vox_o': parts_voxel64, 'part_vox_v': parts_voxel_scaled64})
    
    # with open(os.path.join(pqroot, '../original/data', category + "_info.json"), 'r') as f:
    #     ipdb.set_trace()
    #     f.keys()
    with open(os.path.join(pqroot, category + "_info.json"), 'w') as f:
        json.dump(class_info, f)
    
    match_id = scipy.io.loadmat(os.path.join(pqroot, 'image_split', category, 'img_voxel_idxs.mat'))  # 1-3839, 1-216
    for phase in ['train', 'val', 'test']:
        split = []
        with open(os.path.join(pqroot, 'image_split', category, '{}.txt'.format(phase)), 'r') as f:
            image_names = f.readlines()
        for name in image_names:
            id_img_ori = int(name.split('.')[0])
            img_id_real = list(match_id['img_idxs'][0]).index(id_img_ori)  # 0-3493
            voxel_id_ori = match_id['voxel_idxs'][0, img_id_real]  # 1-216
            voxel_id_real = voxel_id_ori - 1 # 0-215
            model_info = {'model_id': 'none', 'anno_id': str(voxel_id_real)}
            if model_info not in split:
                split.append(model_info)
        with open(os.path.join(pqroot, 'train_val_test_split', category + '.' + phase + '.json'), 'w') as f:
            json.dump(split, f)
        print(phase, len(split))
    print('max_n_parts', max_n_parts)


if __name__ == "__main__":
    category = '3dprnnNightstand'
    # generate_original_part_voxel_pq(category)
    # cd voxelization
    # python rescale_part_vox.py --src /root/model_split/PQ-NET/data/original_parts_voxel/Lamp
    # python rescale_part_vox.py --src /root/model_split/PQ-NET/data/original_parts_voxel/Chair
    # check_original_part_to_box(category)
    # save_mat_for_visualization(category)

    # python process_data/all/code/voxelization.py
    # python rescale_part_vox.py --src /root/model_split/data/all_classes/chair/primset_sem/pqnet_h5
    # scp -r pqnet_h5 data/Chair; scp -r voxeltxt data/image_split/Chair; scp -r img_voxel_idxs.mat data/image_split/Chair;
    save_json_and_mat(category, save_mat=False)
