from skimage import io, transform
from PyEXR import PyEXRImage
import scipy.io
import numpy as np
import math
import time
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import copy
import matplotlib.patches as patches
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import cv2
import pdb


class Projection(object):
    def __init__(self, cls, full_model):
        self.cls = cls
        self.full_model = full_model
        self.root = os.path.abspath('.')
        self.in_dir = os.path.join(self.root, '../input')
        self.out_dir = os.path.join(self.root, '../output', self.cls)
        self.visual_dir_2d = os.path.join(self.root, '../visual', 'projection', self.cls)
        if not os.path.exists(self.visual_dir_2d):
            os.mkdir(self.visual_dir_2d)
        self.match_id = scipy.io.loadmat(os.path.join(self.out_dir, 'img_voxel_idxs.mat'))
        primset_dir = os.path.join(self.out_dir, 'primset_sem/Myprimset_all.mat')
        self.primset = scipy.io.loadmat(primset_dir)['primset']  # 216,1 struct
        voxels_dir = os.path.join(self.out_dir, 'voxels_dir_{}.txt'.format(cls))
        with open(voxels_dir, 'r') as f:
            self.voxels_dir_all = f.readlines()
        self.prim_to_mesh = True
        self.mesh_to_prim = False
        self.visual_3d = False
        self.visual_2d = True
        self.save_2d_kp_dir = os.path.join(self.out_dir, 'kp_2d_projected.json')
        if self.full_model:
            self.save_2d_kp_dir = os.path.join(self.out_dir, 'kp_2d_projected_full.json')
            self.visual_2d = False
        if self.cls[:6] == '3dprnn':
            self.v_size = 30
            self.resize_dim = 64.
            self.img_size = 64
            rot_mat_dir = os.path.join(self.in_dir, '3dprnn/depth_map/rot_mat_{}.mat'.format(self.cls[6:]))
            self.rot_mat_all = scipy.io.loadmat(rot_mat_dir)['rot_mat']
            self.depth_png_dir = os.path.join(self.out_dir, 'images_crop_object')
            if not os.path.exists(self.depth_png_dir):
                os.mkdir(self.depth_png_dir)
        else:
            self.v_size = 32
            self.resize_dim = 320.
            self.visual_dir_3d = os.path.join(self.root, '../visual', 'prim2mesh', self.cls)
            if not os.path.exists(self.visual_dir_3d):
                os.mkdir(self.visual_dir_3d)
            # self.inter_dir = os.path.join(self.out_dir, 'inter')
            # if not os.path.exists(self.inter_dir):
            #     os.mkdir(self.inter_dir)
            pix3d_dir = os.path.join(self.root, '../input/pix3d.json')
            self.pix3d = json.load(open(pix3d_dir, 'r'))
            self.scale_all = {'kp': [], 'prim': []}
            self.keypoints_3d_all = self.load_keypoints_3d()
        self.prim_corners_all = self.primset_to_corners()

    def get_transform_3d(self, ori=None):
        if self.cls[:6] == '3dprnn':
            if self.cls == '3dprnntable' or self.cls == '3dprnnnight_stand':
                z_theta = np.pi / 2
                z_rot = np.array([
                    [np.cos(z_theta), np.sin(z_theta), 0],
                    [-np.sin(z_theta), np.cos(z_theta), 0],
                    [0, 0, 1]
                ])
                return z_rot
            else:
                return np.eye(3)
        # x pi/2, z pi
        if ori == 'kp':
            x_theta = -np.pi / 2
        elif ori == 'prim':
            x_theta = np.pi / 2 #left hand
        else:
            raise NotImplementedError
        x_rot = np.array([
            [1, 0, 0],
            [0, np.cos(x_theta), np.sin(x_theta)],
            [0, -np.sin(x_theta), np.cos(x_theta)]
        ])
        z_theta = np.pi
        z_rot = np.array([
            [np.cos(z_theta), np.sin(z_theta), 0],
            [-np.sin(z_theta), np.cos(z_theta), 0],
            [0, 0, 1]
        ])
        if ori == 'kp':
            transform = np.dot(z_rot, x_rot)
        elif ori == 'prim':
            transform = np.dot(x_rot, z_rot)
        else:
            raise NotImplementedError
        return transform

    def load_keypoints_3d(self):
        keypoints_3d_all = []   # [[[]]], num_model, num_kp, 3
        max_num_kp = 0
        # max_z = 0
        for voxel_dir in self.voxels_dir_all:
            kp_dir = voxel_dir.split('/')[:-1] + ['3d_keypoints.txt']
            kp_dir = os.path.join(self.in_dir, 'model', kp_dir[0], kp_dir[1], kp_dir[2])
            keypoints_3d = []
            with open(kp_dir, 'r') as f:
                kp_file = f.readlines()
                for line in kp_file:
                    kp = line.strip().split(' ')[:3]
                    keypoints_3d.append([float(x) for x in kp])
            keypoints_3d_all.append(keypoints_3d)
            num_kp = len(keypoints_3d)
            # z = np.max(np.abs(np.array(keypoints_3d)[:, 2]))
            if num_kp > max_num_kp:
                max_num_kp = num_kp
            scale = np.max(np.abs(np.array(keypoints_3d)))
            self.scale_all['kp'].append(scale)
            # if z > max_z:
            #     max_z = z
        if self.mesh_to_prim:
            keypoints_3d_all_trans = []
            transform = self.get_transform_3d(ori='kp')
            for model_i in range(len(keypoints_3d_all)):
                keypoints_3d = keypoints_3d_all[model_i]
                for kp_i in range(max_num_kp - len(keypoints_3d)):
                    keypoints_3d.append([0, 0, 0])
                keypoints_3d = np.array(keypoints_3d).T
                keypoints_3d = np.dot(transform, keypoints_3d)
                keypoints_3d = keypoints_3d.T / self.scale_all['kp'][model_i] * 0.5
                keypoints_3d_all_trans.append(keypoints_3d.tolist())
            keypoints_3d_all = keypoints_3d_all_trans
        return keypoints_3d_all

    def combine_ori_sym(self, ori, sym, cls):
        prim_all = copy.deepcopy(ori)
        cls_all = copy.deepcopy(cls)
        for i in range(ori.shape[0]):
            if np.sum(np.abs(sym[i, 10:])) > 0:
                prim_all = np.vstack((prim_all, sym[i:i + 1, :]))
                cls_all = np.hstack((cls_all, cls[i:i + 1]))
        return prim_all, cls_all

    def primset_to_corners(self):
        prim_corners_all = []
        num_obj = self.primset.shape[0]
        for i in range(num_obj):
            prim_ori = self.primset[i, 0]['ori'][0, 0]
            if self.full_model:
                ori = self.primset[i, 0]['ori'][0, 0]
                sym = self.primset[i, 0]['sym'][0, 0]
                cls = self.primset[i, 0]['cls'][0, 0][0]  # start from 1 (will be decreased later in get_label)
                prim_ori, cls_all = self.combine_ori_sym(ori, sym, cls)
            prim_corners = self.prim_all_to_cornerset(i, prim_ori, self.prim_to_mesh)
            prim_corners_all.append(prim_corners)
            if self.visual_3d:
                keypoints_3d = self.keypoints_3d_all[i]
                # print(i, np.min(np.array(prim_corners)[:,:,2]), np.min(np.array(keypoints_3d)[:, 2]),
                #       np.max(np.array(prim_corners)[:,:,2]), np.max(np.array(keypoints_3d)[:, 2]))
                self.visualize_3d_kp(i, prim_corners, keypoints_3d)
                # print(i, np.array(keypoints_3d))
        # save_dir = os.path.join(self.inter_dir, 'prim_corners_all.mat')
        # scipy.io.savemat(save_dir, {'prim_corners_all': np.array(prim_corners_all)})
        return prim_corners_all # [[np (3, 8)]]

    def prim_all_to_cornerset(self, model_i, prim_all, prim_to_mesh):
        cornerset = []  # [np, 8*3]
        for j in range(prim_all.shape[0]):
            prim_r = prim_all[j]
            prim_pt = self.prim_to_corners(model_i, prim_r, prim_to_mesh)
            cornerset.append(prim_pt)
        return cornerset

    def prim_to_corners(self, model_i, prim_r, prim_to_mesh):
        prim_pt_x = np.array([0, prim_r[10], prim_r[10], 0, 0, prim_r[10], prim_r[10], 0])
        prim_pt_y = np.array([0, 0, prim_r[11], prim_r[11], 0, 0, prim_r[11], prim_r[11]])
        prim_pt_z = np.array([0, 0, 0, 0, prim_r[12], prim_r[12], prim_r[12], prim_r[12]])
        prim_pt = [prim_pt_x, prim_pt_y, prim_pt_z]
        prim_pt = np.array(prim_pt)

        prim_pt = prim_pt.T + prim_r[13:16]
        prim_pt_mean = prim_pt.mean(axis=0)

        axis = prim_r[16:19]
        theta = prim_r[19]
        Rrot = self.get_rot_matrix(axis, theta)

        prim_pt -= prim_pt_mean
        prim_pt = prim_pt.dot(Rrot)
        prim_pt += prim_pt_mean
        # if not opt.global_denorm:
        #     prim_pt[np.where(prim_pt > self.v_size)] = self.v_size
        #     prim_pt = prim_pt / self.v_size
        if prim_to_mesh:
            prim_pt = self.prim_to_mesh_coordinate(model_i, prim_pt)
        return prim_pt

    def prim_to_mesh_coordinate(self, model_i, prim_pt):
        prim_pt = prim_pt / self.v_size - 0.5
        if not self.mesh_to_prim:
            transform = self.get_transform_3d(ori='prim')
            prim_pt = np.dot(transform, prim_pt.T).T
            if self.cls[:6] == '3dprnn':
                scale = 0.5##np.max(np.abs(np.array(keypoints_3d)))
            else:
                scale = self.scale_all['kp'][model_i]
            prim_pt = prim_pt / 0.5 * scale

        if self.cls == '3dprnntable':
            prim_pt[:, 1] = -prim_pt[:, 1]
        return prim_pt

    def get_rot_matrix(self, axis, theta):
        vx = self.getVx(axis)
        Rrot = math.cos(theta) * np.eye(3) + math.sin(theta) * vx + \
               (1 - math.cos(theta)) * np.array([axis], ).T * np.array([axis], )
        return Rrot

    def getVx(self, axis):
        vx = np.array([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]])
        return vx

    def visualize_3d_kp(self, model_id, prim_corners, keypoints_3d):
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.clear()
        colors = ['r', 'g', 'b', 'y', 'm', 'c', 'w', 'k', 'r', 'g', 'b', 'y', 'm', 'c', 'w', 'k',
                  'r', 'g', 'b', 'y', 'm', 'c', 'w', 'k', 'r', 'g', 'b', 'y', 'm', 'c', 'w', 'k',
                  'r', 'g', 'b', 'y', 'm', 'c', 'w', 'k', 'r', 'g', 'b', 'y', 'm', 'c', 'w', 'k']
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # for kp in keypoints_3d:
        for kp_i in range(len(keypoints_3d)):
            kp = keypoints_3d[kp_i]
            if kp_i == 0:
                ax.scatter(kp[0], kp[1], kp[2], zdir='z', c='y', marker='.', s=800)
            else:
                ax.scatter(kp[0], kp[1], kp[2], zdir='z', c='r', marker='.', s=400)
        for prim_pt in prim_corners:
            assert prim_pt.shape[0] == 8
            for i in range(prim_pt.shape[0]):
                corner = prim_pt[i]
                ax.scatter(corner[0], corner[1], corner[2], zdir='z', c='g', marker='.', s=400)
        # for i in range(8):
        #   ax.plot([x3d[0,i], x3d[0,i+18]], [x3d[1,i],x3d[1,i+18]], [x3d[2,i],x3d[2,i+18]], c=(0,0,0))
        #   ax.set_xlabel('X Label')
        #   ax.set_ylabel('Y Label')
        #   ax.set_zlabel('Z Label')
        #   for direction in (-1, 1):
        #       for point in np.diag(direction * np.array([0.5,0.3,0.3])):
        #           ax.plot([point[0]], [point[1]], [point[2]], 'w')
        # ax.view_init(elev=105, azim=270)
        plt.draw()
        # plt.waitforbuttonpress()
        save_dir = os.path.join(self.visual_dir_3d, str(model_id)+'.png')
        plt.savefig(save_dir)
        plt.close('all')

    def project_3d_to_2d(self):
        # dict{
        # 'after':{
        # '0':{
        # 'name':'0001.png',
        # 'keypoint': [len = num_prim * 8  [x, y]]}}}
        prim_corners_2d_all = {'after': {}}
        if self.cls[:6] == '3dprnn':
            prim_corners_2d_all = self.project_3d_to_3d_3dprnn(prim_corners_2d_all)
        else:
            prim_corners_2d_all = self.project_3d_to_2d_pix3d(prim_corners_2d_all)
        self.save_kp_json(prim_corners_2d_all)

    def project_3d_to_2d_pix3d(self, prim_corners_2d_all):
        for i in range(len(self.pix3d)):
            if self.pix3d[i]['category'] == self.cls:
                ins = self.pix3d[i]
                img_name = ins['img'].split('/')[-1]
                after_img_name = img_name.split('.')[0] + '.png'
                id_img_ori = int(img_name.split('.')[0])  # 1-3839
                if id_img_ori not in list(self.match_id['img_idxs'][0]):
                    print('*' * 30, 'skip', id_img_ori)
                    continue
                img_id_real = list(self.match_id['img_idxs'][0]).index(id_img_ori)  # 0-3493
                voxel_id_ori = self.match_id['voxel_idxs'][0, img_id_real]  # 1-216
                voxel_id_ori -= 1

                kp_3d = np.array(self.keypoints_3d_all[voxel_id_ori])
                kp_2d_render = self.project_3d_kp_in_mesh_coordinate_one(ins, kp_3d, type='render')
                kp_2d_after = self.project_3d_kp_in_mesh_coordinate_one(ins, kp_3d, type='after')

                prim_corners_3d = np.vstack(self.prim_corners_all[voxel_id_ori])
                prim_corners_2d = self.project_3d_kp_in_mesh_coordinate_one(ins, prim_corners_3d, type='after')
                box2d = self.kp2d_to_box2d(prim_corners_2d)

                if self.visual_2d:
                    image = self.load_image(img_name, after_img_name, type='render')
                    self.visualize_img(img_name, image, kp_2d_render, 'render')
                    image = self.load_image(img_name, after_img_name, type='after')
                    self.visualize_img(img_name, image, kp_2d_after, 'after_kp')
                    self.visualize_img(img_name, image, (prim_corners_2d, box2d), 'after_prim')

                prim_corners_2d_all['after'][str(img_id_real)] = {}
                prim_corners_2d_all['after'][str(img_id_real)]['name'] = after_img_name
                prim_corners_2d_all['after'][str(img_id_real)]['keypoint'] = prim_corners_2d.tolist()
                # print(prim_corners_2d)
                # print(prim_corners_2d_all['after'])
                # pdb.set_trace()
        return prim_corners_2d_all

    def project_3d_to_3d_3dprnn(self, prim_corners_2d_all):
        num = self.match_id['img_idxs'].shape[1]
        count_miss = 0
        for i in range(num):
            img_name = '0000' + str(self.match_id['img_idxs'][0, i])
            img_name = img_name[-4:] + '.exr'  # 1-3839
            # after_img_name = img_name.split('.')[0] + '.png'  # 1-3839
            after_img_name = img_name.split('.')[0] + '.mat'  # 1-3839
            img_id_real = i     # 0-3493
            voxel_id_ori = self.match_id['voxel_idxs'][0, i]  # 1-216
            voxel_id_ori -= 1
            depth_exr_dir = os.path.join(self.in_dir, '3dprnn/depth_map/depth_render_{}/'.format(self.cls[6:]),
                                         img_name)
            depth_png_dir = os.path.join(self.depth_png_dir, after_img_name)
            depth, count_miss, success = self.load_exr(depth_exr_dir, count_miss, i)
            if not success:
                print('=' * 30, 'ERROR')
                pdb.set_trace()
            # cv2.imwrite(depth_png_dir, depth)
            if not self.full_model:
                scipy.io.savemat(depth_png_dir, {'depth': depth})
            ins = {}
            ins['img_size'] = (self.img_size, self.img_size)
            ins['bbox'] = (0, 0, self.img_size, self.img_size)
            ins['trans_mat'] = [0, 0, 1.3]
            ins['rot_mat'] = self.rot_mat_all[i]
            ins['focal_length'] = 35
            # pdb.set_trace()
            prim_corners_3d = np.vstack(self.prim_corners_all[voxel_id_ori])
            prim_corners_2d = self.project_3d_kp_in_mesh_coordinate_one(ins, prim_corners_3d, type='after')
            box2d = self.kp2d_to_box2d(prim_corners_2d)

            if self.visual_2d:
                image = self.load_image(img_name, after_img_name, type='after')
                self.visualize_img(img_name, image, (prim_corners_2d, box2d), 'after_prim')

            prim_corners_2d_all['after'][str(img_id_real)] = {}
            prim_corners_2d_all['after'][str(img_id_real)]['name'] = after_img_name
            prim_corners_2d_all['after'][str(img_id_real)]['keypoint'] = prim_corners_2d.tolist()
        return prim_corners_2d_all

    def kp2d_to_box2d(self, kp_2d):
        prim_num = kp_2d.shape[0] // 8
        prim_box_2d = np.zeros((prim_num, 4))
        for i in range(prim_num):
            prim_kp_i = kp_2d[i * 8: i * 8 + 8]
            min_x, min_y = np.min(prim_kp_i, axis=0)
            max_x, max_y = np.max(prim_kp_i, axis=0)
            prim_box_2d[i, :] = np.array([min_x, min_y, max_x, max_y])
        return prim_box_2d

    def project_3d_kp_in_mesh_coordinate_one(self, ins, kp_3d, type=None):
        kp_2d = self.project_one_obj(ins, kp_3d, type)
        # pdb.set_trace()
        if type == 'after':
            kp_2d = self.adjust_2d_kp_for_crop(ins, kp_2d)
        return kp_2d

    def adjust_2d_kp_for_crop(self, ins, kp_2d):
        thresh = 0.1
        w, h = ins['img_size']
        bbox = ins['bbox']
        x_min, y_min, x_max, y_max = bbox
        x_min = max(0, x_min - (x_max - x_min) * thresh)
        x_max = min(w, x_max + (x_max - x_min) * thresh)
        y_min = max(0, y_min - (y_max - y_min) * thresh)
        y_max = min(h, y_max + (y_max - y_min) * thresh)
        w_crop = x_max - x_min
        h_crop = y_max - y_min
        short_edge = min(w_crop, h_crop)
        scale = self.resize_dim / short_edge
        # w_resize = int(w_crop * resize_dim / short_edge)
        # h_resize = int(h_crop * resize_dim / short_edge)
        kp_2d[:, 0] -= x_min
        kp_2d[:, 1] -= y_min
        kp_2d *= scale
        return kp_2d

    def project_one_obj(self, ins, kp_3d, type=None):
        kp_3d = kp_3d.T
        kp_3d = np.vstack((kp_3d, np.ones((1, kp_3d.shape[1]))))
        RT = self.get_obj_RT(ins)
        K = self.get_calibration_matrix_K(ins, type)
        kp_3d = np.dot(RT, kp_3d)
        kp_2d = np.dot(K, kp_3d)
        kp_2d = kp_2d / kp_2d[2:, :]
        kp_2d = kp_2d[:2, :].T
        return kp_2d

    def get_obj_RT(self, ins):
        rot_mat = ins['rot_mat']
        trans_mat = ins['trans_mat']
        rot_mat = np.array(rot_mat)
        trans_mat = np.array(trans_mat).reshape(-1, 1)
        # rot_mat = np.eye(3)
        # trans_mat = np.array([0.2,0.2,trans_mat[2,0]]).reshape(-1, 1)
        RT = np.hstack((rot_mat, trans_mat))
        return RT

    def get_calibration_matrix_K(self, ins, type=None):
        f_in_mm = ins['focal_length']
        if type == 'render':
            w, h = ins['img_size']
            while w > 400 or h > 400:
                if w / 2.0 == int(w / 2.0) and h / 2.0 == int(h / 2.0):
                    w = w / 2.0
                    h = h / 2.0
                    # print(w, h)
                else:
                    break
            img_width, img_height = w, h
        else:
            img_width, img_height = ins['img_size']
        # orientation = 'HORIZONTAL'
        # scale = 1   # scene.render.resolution_percentage / 100
        # pixel_aspect_ratio = 1  # scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        # aspect_ratio = float(img_width) / float(img_height)
        sensor_width_in_mm = 32
        # sensor_height_in_mm = 18
        # pixel_width = sensor_width_in_mm / img_width
        # pixel_height = sensor_height_in_mm / img_height
        # fx = f_in_mm / pixel_width
        # fy = f_in_mm / pixel_height# * aspect_ratio * aspect_ratio
        fx = f_in_mm * img_width / sensor_width_in_mm
        fy = f_in_mm * img_width / sensor_width_in_mm
        # principal_point_x = sensor_width_in_mm / 2
        # principal_point_y = sensor_height_in_mm / 2
        # cx = principal_point_x / pixel_width    #==img_width / 2
        # cy = principal_point_y / pixel_height   #==img_height / 2
        # scale = 1 / sensor_width_in_mm##0.03#0.032?
        # fx = f_in_mm * img_width * scale
        # fy = f_in_mm * img_height * scale * aspect_ratio
        cx = img_width / 2
        cy = img_height / 2
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        # rotate pi along axis z, equals to -x, -y
        R_default = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        K = np.dot(K, R_default)
        return K

    def get_camera_RT(self, cam):
        import bpy
        from mathutils import Matrix
        R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

        location, rotation = cam.matrix_world.decompose()[0:2]
        R_world2bcam = rotation.to_matrix().transposed()
        T_world2bcam = -1 * np.dot(R_world2bcam, location)
        R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
        T_world2cv = np.dot(R_bcam2cv, T_world2bcam)
        RT = Matrix((R_world2cv[0][:] + (T_world2cv[0],),
                     R_world2cv[1][:] + (T_world2cv[1],),
                     R_world2cv[2][:] + (T_world2cv[2],),))
        R = [R_world2cv[0][0], R_world2cv[0][1], R_world2cv[0][2],
             R_world2cv[1][0], R_world2cv[1][1], R_world2cv[1][2],
             R_world2cv[2][0], R_world2cv[2][1], R_world2cv[2][2]]
        T = T_world2cv

    def save_kp_json(self, kp_2d_all):
        with open(self.save_2d_kp_dir, 'w') as f:
            json.dump(kp_2d_all, f)

    def visualize_img(self, img_name, image, label, type=None):
        fig = plt.figure()
        ax = plt.subplot(1, 1, 1)
        plt.imshow(image)
        if label is not None:
            if type == 'after_prim':
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
            plt.scatter(label[0, 0], label[0, 1], s=100, marker='.', c='g')
            plt.scatter(label[1, 0], label[1, 1], s=100, marker='.', c='r')
            plt.scatter(label[2, 0], label[2, 1], s=100, marker='.', c='b')
            plt.scatter(label[3:, 0], label[3:, 1], s=100, marker='.', c='r')
        save_dir = os.path.join(self.visual_dir_2d, type)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_name = os.path.join(save_dir, img_name.split('.')[0] + '.png')
        plt.savefig(save_name)
        plt.close('all')

    def load_image(self, img_name, after_img_name, type=None):
        if type == 'before':
            img_dir = os.path.join(self.in_dir, 'img', self.cls, img_name)
        elif type == 'after':
            img_dir = os.path.join(self.out_dir, 'images_crop_object', after_img_name)
        elif type == 'render':
            img_dir = os.path.join(self.root, '../input/depth_map/render_{}'.format(self.cls),
                                   'model_{}.png'.format(int(img_name.split('.')[0])-1))
        else:
            raise NotImplementedError
        print(img_name, img_dir)
        if after_img_name[-4:] == '.mat':
            image = scipy.io.loadmat(img_dir)['depth']
        else:
            image = cv2.imread(img_dir)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_exr(self, depth_exr_dir, count_miss, i):
        rgb_img = PyEXRImage(depth_exr_dir)
        rgb = np.reshape(np.array(rgb_img, copy=False), (rgb_img.height, rgb_img.width, 4))
        assert (rgb[:, :, 0] == rgb[:, :, 1]).all() and (rgb[:, :, 1] == rgb[:, :, 2]).all()
        if np.sum(rgb[:, :, 0] < 0) > 10 or abs(np.sum(rgb[:, :, 0] > 1) - np.sum(rgb[:, :, 0] == 255)) > 10:
            count_miss += 1
            return None, None, False
        # rgb[:, :, 0] = rgb[:, :, 0] - 255 * (rgb[:, :, 0] == 255)
        rgb[rgb < 0] = 0.
        rgb[rgb > 1] = 0.
        assert np.sum(rgb[:, :, 0]) > 100.
        assert np.sum(rgb[:, :, 0] > 1) == 0 and np.sum(rgb[:, :, 0] < 0) == 0
        rgb_label = rgb[:, :, 0]
        x, y = (rgb_label > 0.).nonzero()
        rgb_label = rgb_label[np.min(x): np.max(x) + 1, np.min(y): np.max(y) + 1]
        h, w = rgb_label.shape
        if h < w:
            pad = (w - h) // 2
            tmp_label = np.zeros((w, w))
            tmp_label[pad: pad + h, :] = copy.deepcopy(rgb_label)
        elif h > w:
            pad = (h - w) // 2
            tmp_label = np.zeros((h, h))
            tmp_label[:, pad: pad + w] = copy.deepcopy(rgb_label)
        else:
            tmp_label = copy.deepcopy(rgb_label)
        tmp_label = transform.resize(tmp_label, (self.img_size, self.img_size), mode='constant', anti_aliasing=True)
        # if i % 5 == 0:
        #     ff = plt.figure()
        #     plt.imshow(tmp_label)
        #     plt.show()
        #     # fff = plt.figure()
        #     # plt.imshow(rgb[:, :, 0])
        #     # plt.show()
        #     import pdb
        #     pdb.set_trace()
        return tmp_label, count_miss, True


if __name__ == '__main__':
    cls_all = ['chair', 'bed', 'bookcase', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
    cls = '3dprnntable'
    full_model = True
    root = os.path.abspath('.')
    visual_dir_2d = os.path.join(root, '../visual', 'projection')
    if not os.path.exists(visual_dir_2d):
        os.mkdir(visual_dir_2d)
    if cls[:6] != '3dprnn':
        visual_dir_3d = os.path.join(root, '../visual', 'prim2mesh')
        if not os.path.exists(visual_dir_3d):
            os.mkdir(visual_dir_3d)
    proj = Projection(cls, full_model)
    proj.project_3d_to_2d()
