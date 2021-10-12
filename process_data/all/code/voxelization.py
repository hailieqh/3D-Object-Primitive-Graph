import os
import scipy.io
import numpy as np
import copy
import math
import h5py


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


def check_voxel_in_box(voxel, voxel_coor, shape, trans, Rv, theta, cen_mean,voxel_scale):
    cnt = 0
    # voxel = np.array([1,1,1])
    # cen_mean = np.array([14.343102,22.324961,23.012661])
    in_box = 0
    voxel_coor = voxel_coor - 0.5#旋转前voxel的中心坐标
    voxel_coor = voxel_coor - cen_mean #平移之后voxel的坐标
    theta = -theta
    vx = getVx(Rv)
    Rrot = math.cos(theta) * np.eye(3) + math.sin(theta) * vx + (1 - math.cos(theta)) * np.array([Rv],).T * np.array([Rv],)
    voxel_coor = voxel_coor.dot(Rrot) #将voxel倒着旋转 voxel为新的坐标系下的坐标
    voxel_coor = voxel_coor + cen_mean
    voxel_coor = voxel_coor - trans #新的坐标原点位于primitive的最小处
    thresh = 0
    lbound = 0.5
    rbound = 0.1
    # if (voxel[:,0] >= -lbound or np.abs(voxel[:,0]) < thresh) and (voxel[:,0] <= (shape[0] - rbound) or np.abs(voxel[:,0] - shape[0]) < thresh):
    #     if (voxel[:,1] >= -lbound or np.abs(voxel[:,1]) < thresh) and (voxel[:,1] <= (shape[1] - rbound) or np.abs(voxel[:,1] - shape[1]) < thresh):
    #         if (voxel[:,2] >= -lbound or np.abs(voxel[:,2]) < thresh) and (voxel[:,2] <= (shape[2] - rbound) or np.abs(voxel[:,2] - shape[2]) < thresh):
    #             in_box = 1

    vox_a = ((voxel_coor[:, 0] >= -lbound) | (np.abs(voxel_coor[:, 0]) < thresh)) & ((voxel_coor[:, 0] <= (shape[0] - rbound)) | (np.abs(voxel_coor[:, 0] - shape[0]) < thresh))
    vox_b = ((voxel_coor[:, 1] >= -lbound) | (np.abs(voxel_coor[:, 1]) < thresh)) & ((voxel_coor[:, 1] <= (shape[1] - rbound)) | (np.abs(voxel_coor[:, 1] - shape[1]) < thresh))
    vox_c = ((voxel_coor[:, 2] >= -lbound) | (np.abs(voxel_coor[:, 2]) < thresh)) & ((voxel_coor[:, 2] <= (shape[2] - rbound)) | (np.abs(voxel_coor[:, 2] - shape[2]) < thresh))
    voxel_in_prim = np.zeros(voxel.shape[0]) #27000 * 1
    #voxel_in_prim = (voxel[:, 0]) & (voxel[:, 1]) & (voxel[:, 2])
    voxel_in_prim = vox_a & vox_b & vox_c

    for i in range(voxel_in_prim.shape[0]): #27000
        if voxel_in_prim[i]:
            voxel_in_prim[i] = 1
            cnt+=1
    voxel_in_prim = voxel_in_prim.reshape(voxel_scale, voxel_scale, voxel_scale)
    voxel += voxel_in_prim
    voxel = (voxel > 0).astype(np.float32)
    return voxel


def prim_to_voxel(voxel, prim_r, voxel_scale, aabb=False):
    cnt = 0
    voxel_one = np.zeros([27001, 1])
    shape = prim_r[10:13]
    trans = prim_r[13:16]
    Rv = prim_r[16:19]
    Rv = np.array(Rv, dtype=int)
    theta = prim_r[19]
    scale = [1, 1, 1]
    vx = getVx(Rv)
    # Rrot = math.cos(theta)*np.eye(3) + math.sin(theta)*vx + (1-math.cos(theta)) * Rv.T * Rv
    Rrot = math.cos(theta) * np.eye(3) + math.sin(theta) * vx + (1 - math.cos(theta)) * np.array([Rv], ).T * np.array(
        [Rv], )
    cen_mean = get_mean(shape, trans, scale, Rrot)

    # in_box = np.zeros([voxel_scale, voxel_scale, voxel_scale])
    # voxel_coor = np.zeros([voxel_scale, voxel_scale, voxel_scale, 3])  # （30，30，30，3） matrix of voxel's coordinate
    # import pdb;pdb.set_trace()
    # for vx in range(1, voxel_scale + 1):  # 1-30
    #     for vy in range(1, voxel_scale + 1):
    #         for vz in range(1, voxel_scale + 1):
    #             voxel_coor[vx - 1, vy - 1, vz - 1, 0:3] = [vx, vy, vz]
    # voxel_coor = voxel_coor.reshape(-1, 3)
    vx = np.arange(voxel_scale)
    vy = np.arange(voxel_scale)
    vz = np.arange(voxel_scale)
    vx, vy, vz = np.meshgrid(vx, vy, vz)
    vx = np.reshape(vx, -1)
    vy = np.reshape(vy, -1)
    vz = np.reshape(vz, -1)
    voxel_coor = np.vstack((vy, vx, vz)).T + 1
    # pdb.set_trace()
    voxel = check_voxel_in_box(voxel, voxel_coor, shape, trans, Rv, theta, cen_mean, voxel_scale)
    if aabb:
        nonzeros = np.nonzero(voxel)
        if nonzeros[0].shape[0] > 0:
            x0, x1 = nonzeros[0].min(), nonzeros[0].max() + 1
            y0, y1 = nonzeros[1].min(), nonzeros[1].max() + 1
            z0, z1 = nonzeros[2].min(), nonzeros[2].max() + 1
            voxel[x0:x1, y0:y1, z0:z1] = 1
        else:
            print('skip too thin')
    return voxel


def prim_to_vox_one_class_accel_primset(i, primset, voxel_scale, aabb=False):
    vox_ori = primset[i, 0]['ori'][0, 0]
    vox_sym = primset[i, 0]['sym'][0, 0]
    vox_cls = primset[i, 0]['cls'][0, 0] - 1  # start from 0
    # print(vox_cls)
    prim_num = vox_ori.shape[0]
    prim_sum = np.sum(np.sum(vox_sym[:, 10:], axis=1) > 0) + prim_num
    voxel_sum = np.zeros([voxel_scale, voxel_scale, voxel_scale])
    voxel_idv = np.zeros([prim_sum, voxel_scale, voxel_scale, voxel_scale])
    prims = np.zeros([prim_sum, 20])
    cls = np.zeros([prim_sum])
    i_idv = 0
    for j in range(prim_num):
        # if i == 132 and j == 3:
        #     print(j, 'skip too thin')
        #     import ipdb; ipdb.set_trace()
        prim_r = vox_ori[j, :]
        voxel = np.zeros([voxel_scale, voxel_scale, voxel_scale])
        voxel = prim_to_voxel(voxel, prim_r, voxel_scale, aabb)
        voxel_sum += voxel
        voxel_idv[i_idv, :, :, :] = copy.deepcopy(voxel)
        prims[i_idv, :] = copy.deepcopy(prim_r)
        cls[i_idv] = copy.deepcopy(vox_cls[0, j])
        i_idv += 1
        prim_r = vox_sym[j, :]
        if np.sum(np.abs(prim_r[10:])) > 0:
            voxel = np.zeros([voxel_scale, voxel_scale, voxel_scale])
            voxel = prim_to_voxel(voxel, prim_r, voxel_scale, aabb)
            voxel_sum += voxel
            voxel_idv[i_idv, :, :, :] = copy.deepcopy(voxel)
            prims[i_idv, :] = copy.deepcopy(prim_r)
            cls[i_idv] = copy.deepcopy(vox_cls[0, j])
            i_idv += 1
    # print(prim_sum, i_idv)
    voxel_sum = (voxel_sum > 0).astype(np.float32)
    return voxel_sum, voxel_idv, prims, cls


def voxelize_primset(cls, voxel_scale, aabb=False):
    root = os.path.abspath('.')
    out_dir = os.path.join(root, '../output', cls, 'primset_sem')
    primset_dir = os.path.join(out_dir, 'Myprimset_all.mat')
    if aabb:
        voxel_dir = os.path.join(out_dir, 'Myvoxel_all_aabb.mat')
    else:
        voxel_dir = os.path.join(out_dir, 'Myvoxel_all.mat')
    primset = scipy.io.loadmat(primset_dir)['primset']
    num = primset.shape[0]
    voxel_all = np.zeros([num, voxel_scale, voxel_scale, voxel_scale])
    for i in range(num):
        voxel_i, _, _, _ = prim_to_vox_one_class_accel_primset(i, primset, voxel_scale, aabb)
        voxel_all[i, :, :, :] = voxel_i
        if i % 10 == 0:
            print(i)
    scipy.io.savemat(voxel_dir, {'voxTile': voxel_all})


def voxelize_primset_h5(cls, voxel_scale, aabb=False):
    # root = os.path.abspath('.')
    # out_dir = os.path.join(root, '../output', cls, 'primset_sem')
    out_dir = os.path.join('/root/model_split/data/all_classes', cls, 'primset_sem')
    primset_dir = os.path.join(out_dir, 'Myprimset_all.mat')
    save_dir = os.path.join(out_dir, 'pqnet_h5')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if aabb:
        voxel_dir = os.path.join(out_dir, 'Myvoxel_all_aabb.mat')
    primset = scipy.io.loadmat(primset_dir)['primset']
    num = primset.shape[0]
    voxel_all = np.zeros([num, voxel_scale, voxel_scale, voxel_scale])
    for i in range(num):
        voxel_i, voxel_idv_i, prims_i, cls_i = prim_to_vox_one_class_accel_primset(i, primset, voxel_scale, aabb)
        voxel_all[i, :, :, :] = voxel_i
        n_parts = voxel_idv_i.shape[0]
        vox_dim = voxel_idv_i.shape[-1]
        if i % 10 == 0:
            print(i)
        save_dir_i = os.path.join(save_dir, '{}.h5'.format(i))
        with h5py.File(save_dir_i, 'w') as fv:
            fv.create_dataset('shape_voxel', shape=(vox_dim, vox_dim, vox_dim),
                dtype=np.bool, data=voxel_i.astype(np.bool), compression=9)
            fv.create_dataset('parts_voxel64', shape=(n_parts, vox_dim, vox_dim, vox_dim),
                dtype=np.bool, data=voxel_idv_i.astype(np.bool), compression=9)
            fv.create_dataset('prims', shape=(n_parts, 20),
                dtype=np.float, data=prims_i.astype(np.float), compression=9)
            fv.create_dataset('cls', shape=(n_parts, 1),
                dtype=np.uint8, data=cls_i.astype(np.uint8), compression=9)
            fv.attrs['n_parts'] = n_parts
    if aabb:
        scipy.io.savemat(voxel_dir, {'voxTile': voxel_all})


if __name__ == '__main__':
    cls_all = ['chair', 'bed', 'bookcase', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
    cls = '3dprnnnight_stand'
    if cls[:6] == '3dprnn':
        voxel_scale = 30
    else:
        voxel_scale = 32
    # voxelize_primset(cls, voxel_scale)
    # voxelize_primset(cls, voxel_scale, aabb=True)
    voxelize_primset_h5(cls, voxel_scale, aabb=True)
    # for c in cls_all:
    #     voxelize_primset(c)
