import scipy.io
# import scipy.io as io
import math
import copy
import time
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
from lib.opts import *
from lib.utils.prnn_utils import *
# from lib.datasets.prnn_dataset import load_names, load_prim_points_2d, get_prim_box_2d, load_image
# from lib.datasets.prnn_dataset import transform_image_label, visualize_img
from lib.datasets.bbox_dataset import PadSquare, Rescale, ToTensor
from torchvision import transforms


class Eval(object):
    def __init__(self):
        global opt
        opt = get_opt()
        self.hausdorff_thresh = 0.2
        self.root_dir = opt.data_dir
        self.obj_class = opt.obj_class
        if opt.encoder in ['resnet', 'hg', 'depth_new']:
            # self.label_info = json.load(open(os.path.join(self.root_dir, 'pix3d.json', 'r')))
            self.image_names = None
            # self.bbox_2d = self.load_bbox_2d('bboxes.mat')
            # self.labels = load_labels_from_t7(self.phase, self.root_dir, intervals)
            self.match_id = load_model_id(self.root_dir, self.obj_class)
            transformer = [PadSquare(), Rescale((opt.input_res, opt.input_res)), ToTensor()]
            self.transform = transforms.Compose(transformer)

    def get_test_class(self, phase, i):
        # test_classes = ['test_chair', 'test_table', 'test_night_stand']
        test_classes = ['test_'+x for x in opt.file_names['obj_classes']]
        intervals = opt.intervals
        if i < intervals[phase][0]:
            test_class = test_classes[0]
        elif i < intervals[phase][1]:
            test_class = test_classes[1]
        elif i < intervals[phase][2]:
            test_class = test_classes[2]
        else:
            raise NotImplementedError
        return test_class

    def gt_part(self, pred_cnt, pred_idv, pred_prims, pred_cls, gt_cnt, gt_idv, gt_prims, gt_cls, cnt):
        thresh = 1.
        part_id = int(opt.revise_mode[-1])
        iou_score = np.sum(np.logical_and(gt_cnt, pred_cnt)) / np.sum(np.logical_or(gt_cnt, pred_cnt))
        if iou_score < thresh:
            pred_part_id = [ind for (ind, x) in enumerate(pred_cls) if x == part_id]
            gt_part_id = [ind for (ind, x) in enumerate(gt_cls) if x == part_id]
            v1 = np.sum(pred_idv[pred_part_id, :, :, :], axis=0)
            v2 = np.sum(gt_idv[gt_part_id, :, :, :], axis=0)
            v1 = (v1 > 0).astype(np.float32)
            v2 = (v2 > 0).astype(np.float32)
            if np.sum(v1) != 0 and np.sum(v2) != 0:
                iou_cushion = np.sum(np.logical_and(v1, v2)) / np.sum(np.logical_or(v1, v2))
                print(iou_score, iou_cushion)
            else:
                print(iou_score)
            pred_idv = np.delete(pred_idv, pred_part_id, axis=0)
            pred_idv = np.concatenate((pred_idv, gt_idv[gt_part_id, :, :, :]), axis=0)
            pred_cnt = np.sum(pred_idv, axis=0)
            pred_cnt = (pred_cnt > 0).astype(np.float32)
            pred_prims = np.delete(pred_prims, pred_part_id, axis=0)
            pred_prims = np.concatenate((pred_prims, gt_prims[gt_part_id, :]), axis=0)
            pred_cls = np.delete(pred_cls, pred_part_id, axis=0)
            pred_cls = np.concatenate((pred_cls, gt_cls[gt_part_id]), axis=0)
        return pred_cnt, pred_idv, pred_prims, pred_cls

    def fix_cushion(self, pred_cnt, pred_idv, pred_prims, pred_cls, gt_cnt, gt_idv, gt_prims, gt_cls, cnt):
        thresh = 1.
        iou_score = np.sum(np.logical_and(gt_cnt, pred_cnt)) / np.sum(np.logical_or(gt_cnt, pred_cnt))
        if iou_score < thresh:
            pred_cushion_id = [ind for (ind, x) in enumerate(pred_cls) if x == 1]
            pred_up_id = [ind for (ind, x) in enumerate(pred_cls) if x == 0 or x == 3]
            pred_down_id = [ind for (ind, x) in enumerate(pred_cls) if x == 2]
            gt_cushion_id = [ind for (ind, x) in enumerate(gt_cls) if x == 1]
            if len(pred_cushion_id) > 0:
                if len(gt_cushion_id) > 1:
                    print(cnt, gt_cushion_id)
                else:
                    pred_z = np.min(pred_prims[pred_cushion_id, 15])
                    gt_z = np.min(gt_prims[gt_cushion_id, 15])
                    delta = gt_z - pred_z
                    pred_prims[pred_cushion_id, 15] += delta
                    pred_prims[pred_up_id, 15] += delta
                    pred_prims[pred_up_id, 12] -= delta
                    pred_prims[pred_down_id, 12] += delta
        return pred_prims

    def match_pred_gt(self, pred_idv, gt_idv):
        thresh = 0.5
        match_ids = []
        for i in range(pred_idv.shape[0]):
            max_iou = 0.
            match_id = -1
            for j in range(gt_idv.shape[0]):
                v1 = pred_idv[i]
                v2 = gt_idv[j]
                iou_score = np.sum(np.logical_and(v1, v2)) / np.sum(np.logical_or(v1, v2))
                if iou_score > thresh and iou_score > max_iou:
                    max_iou = iou_score
                    match_id = j
            if match_id != -1:
                match_ids.append((i, match_id))
        return match_ids

    def fix_extension(self, pred_cnt, pred_idv, pred_prims, pred_cls, gt_cnt, gt_idv, gt_prims, gt_cls, cnt):
        thresh = 1.
        iou_score = np.sum(np.logical_and(gt_cnt, pred_cnt)) / np.sum(np.logical_or(gt_cnt, pred_cnt))
        if iou_score < thresh:
            for i in range(pred_prims.shape[0]):
                # corners = self.prim_to_corners(pred_prims[i, :])
                pred_back_id = [ind for (ind, x) in enumerate(pred_cls) if x == 0]
                pred_cushion_id = [ind for (ind, x) in enumerate(pred_cls) if x == 1]
                pred_leg_id = [ind for (ind, x) in enumerate(pred_cls) if x == 2]
                pred_hand_id = [ind for (ind, x) in enumerate(pred_cls) if x == 3]
                pred_row_id = [ind for (ind, x) in enumerate(pred_cls) if x == 4]
                pred_swivel_id = [ind for (ind, x) in enumerate(pred_cls) if x == 5]
                if len(pred_cushion_id) > 0:
                    cushion_z = np.min(pred_prims[pred_cushion_id, 15])
                    cushion_z_h = np.min(pred_prims[pred_cushion_id, 12])
                    if len(pred_back_id) > 0:
                        back_z = np.min(pred_prims[pred_back_id, 15])
                        back_z_h = np.min(pred_prims[pred_back_id, 12])
                        back_delta = cushion_z + cushion_z_h / 2 - back_z
                        pred_prims[pred_back_id, 15] += back_delta
                        pred_prims[pred_back_id, 12] -= back_delta
                    if len(pred_leg_id) > 0:
                        leg_z = np.min(pred_prims[pred_leg_id, 15])
                        leg_z_h = np.min(pred_prims[pred_leg_id, 12])
                        leg_delta = cushion_z + cushion_z_h / 2 - leg_z - leg_z_h
                        pred_prims[pred_leg_id, 12] += leg_delta
                    if len(pred_hand_id) > 0:
                        hand_z = np.min(pred_prims[pred_hand_id, 15])
                        hand_z_h = np.min(pred_prims[pred_hand_id, 12])
                        hand_delta = cushion_z + cushion_z_h / 2 - hand_z
                        pred_prims[pred_hand_id, 15] += hand_delta
                        pred_prims[pred_hand_id, 12] -= hand_delta
        return pred_prims

    def cal_iou_one_class_one_ins(self, obj_class=None):
        assert opt.encoder in ['resnet', 'hg', 'depth_new']
        # if obj_class is None:
        #     obj_class = opt.obj_class
        self.image_names = load_names('test_' + obj_class, self.root_dir, obj_class)
        save_iou_file = opt.init_source + '/iou_{}_{}.mat'.format(obj_class, opt.eval_mode)
        save_prims_file = opt.init_source + '/prims_{}_{}.mat'.format(obj_class, opt.eval_mode)
        iou = []
        gt_from_prim = None
        primset = None
        if opt.eval_mode == 'ins_voxel':
            gt_dir = self.root_dir + '/ModelNet10_mesh/{}/modelnet_{}_test.mat'.format(obj_class, obj_class)
            gt = scipy.io.loadmat(gt_dir)['voxTile']
            if opt.encoder in ['resnet', 'hg', 'depth_new']:
                gt_from_prim_dir = os.path.join(self.root_dir, opt.file_names['voxel'])
            else:
                gt_from_prim_dir = self.root_dir + '/ModelNet10_mesh/{}/Myvoxset_{}.mat'.format(obj_class, obj_class)
            gt_from_prim = scipy.io.loadmat(gt_from_prim_dir)['voxTile']
        elif opt.eval_mode == 'ins_primset':
            primset = scipy.io.loadmat(os.path.join(self.root_dir, opt.file_names['primset']))
            primset = primset['primset']
        else:
            raise NotImplementedError
        prim_2_vox = Prim2VoxAccel()
        res = scipy.io.loadmat(opt.init_source + '/test_res_mn_{}.mat'.format(obj_class))
        res_x = res['x']
        voxel_length = res_x.shape[0] // 4
        print(voxel_length)
        prims_revised = np.zeros((voxel_length, 20, 20))
        for cnt in range(voxel_length):
            image_name = self.image_names[cnt]
            id_img_ori = int(image_name[0:4])
            cnt_gt = get_model_id(id_img_ori, self.match_id)  # 0-215   # pick out of 216
            if opt.eval_mode == 'ins_voxel':
                gt_cnt = gt_from_prim[cnt_gt]
                gt_idv, gt_prims, gt_cls = None, None, None
            elif opt.eval_mode == 'ins_primset':
                gt_cnt, gt_idv, gt_prims, gt_cls = prim_2_vox.prim_to_vox_one_class_accel_primset(cnt_gt, obj_class, primset)
            else:
                raise NotImplementedError
            pred_cnt, pred_idv, pred_prims, pred_cls = prim_2_vox.prim_to_vox_one_class_accel_one_ins(cnt, obj_class, res)
            if opt.revise_mode in ['gt_part_0', 'gt_part_1', 'gt_part_2', 'gt_part_3', 'gt_part_4', 'gt_part_5']:
                pred_cnt, pred_idv, pred_prims, pred_cls = self.gt_part(pred_cnt, pred_idv, pred_prims, pred_cls,
                                                                           gt_cnt, gt_idv, gt_prims, gt_cls, cnt)
            if opt.revise_mode in ['fix_cushion', 'fix_cu_extension']:
                pred_prims = self.fix_cushion(pred_cnt, pred_idv, pred_prims, pred_cls,
                                              gt_cnt, gt_idv, gt_prims, gt_cls, cnt)
                pred_cnt, pred_idv = prim_2_vox.prim_to_vox_one_class_accel_prims(cnt, obj_class, pred_prims)
            if opt.revise_mode in ['fix_extension', 'fix_cu_extension']:
                pred_prims = self.fix_extension(pred_cnt, pred_idv, pred_prims, pred_cls,
                                              gt_cnt, gt_idv, gt_prims, gt_cls, cnt)
                pred_cnt, pred_idv = prim_2_vox.prim_to_vox_one_class_accel_prims(cnt, obj_class, pred_prims)
            prims_revised[cnt, :pred_prims.shape[0], :] = copy.deepcopy(pred_prims)
            iou_score = np.sum(np.logical_and(gt_cnt, pred_cnt)) / np.sum(np.logical_or(gt_cnt, pred_cnt))
            if np.isnan(iou_score):
                print("Warning: found empty voxel pairs at cnt %f" % cnt)
            else:
                iou.append(iou_score)
            if np.mod(cnt, 10*opt.test_gap) == 0:
                print(cnt)
        iou = np.array(iou)
        print('iou', obj_class, np.mean(iou), iou.shape)
        scipy.io.savemat(save_iou_file, {'iou_{}'.format(obj_class): iou[:, np.newaxis],
                                         'mean_iou_{}'.format(obj_class): np.mean(iou)})
        scipy.io.savemat(save_prims_file, {'prims': prims_revised})
        return np.mean(iou)

    def cal_iou_one_class_by_vox(self, obj_class=None):
        t7_dir = os.path.join(self.root_dir, 'gt_pth')
        # if obj_class is None:
        #     obj_class = opt.obj_class
        self.image_names = load_names('test_'+obj_class, self.root_dir, obj_class)
        save_iou_file = opt.init_source + '/nn_iou_{}.mat'.format(obj_class)
        iou = []
        source = 'prob'
        gt_dir = self.root_dir + '/ModelNet10_mesh/{}/modelnet_{}_test.mat'.format(obj_class, obj_class)
        gt = scipy.io.loadmat(gt_dir)['voxTile']
        if opt.encoder in ['resnet', 'hg', 'depth_new']:
            gt_from_prim_dir = os.path.join(self.root_dir, opt.file_names['voxel'])
        else:
            gt_from_prim_dir = self.root_dir + '/ModelNet10_mesh/{}/Myvoxset_{}.mat'.format(obj_class, obj_class)
        gt_from_prim = scipy.io.loadmat(gt_from_prim_dir)['voxTile']
        pred_id_dir = opt.init_source + '/test_NNfeat_mn_{}_l2.mat'.format(obj_class)
        pred_init_id = scipy.io.loadmat(pred_id_dir)['test_ret_num']
        voxel_length = pred_init_id.shape[0]    # 5000
        print(voxel_length)
        for cnt in range(0, voxel_length, opt.test_gap):
            nn_id = int(pred_init_id[cnt][0])
            assert nn_id >= 0
            # if opt.rgb_con:
            if opt.encoder in ['resnet', 'hg', 'depth_new']:
                # voxel1 = gt[nn_id]
                voxel1 = gt_from_prim[nn_id]
                image_name = self.image_names[cnt]
                id_img_ori = int(image_name[0:4])
                cnt_gt = get_model_id(id_img_ori, self.match_id)    # 0-215   # pick out of 216
                voxel2 = gt_from_prim[cnt_gt]
                # voxel1 = gt_from_prim[cnt_gt]   # upper bound
            else:
                if nn_id < 5555:
                    phase = 'train'
                else:
                    nn_id -= 5555
                    phase = 'val'
                # import pdb
                # pdb.set_trace()
                find_class = self.get_test_class(phase, nn_id)
                print(obj_class, find_class[5:])
                voxel1_name = os.path.join(t7_dir + '/vox', phase, 'voxel{}.mat'.format(nn_id))
                voxel1 = scipy.io.loadmat(voxel1_name)['voxTile']
                if source == 'gt':
                    cnt_gt = cnt
                else:
                    cnt_gt = cnt // 50  # // opt.test_gap
                voxel2 = gt[cnt_gt]
                if not (obj_class == 'table' and cnt_gt == 16):
                    if not obj_class == 'chair':
                        voxel2 = np.rot90(voxel2, 1, (0, 1))
            iou_score = np.sum(np.logical_and(voxel1, voxel2)) / np.sum(np.logical_or(voxel1, voxel2))
            if np.isnan(iou_score):
                iou_score = 0.
                print("Warning: found empty voxel pairs at cnt %f" % cnt)
            iou.append(iou_score)
        iou = np.array(iou)
        print('iou', obj_class, np.mean(iou))
        scipy.io.savemat(save_iou_file, {'iou_{}'.format(obj_class): iou[:, np.newaxis],
                                         'mean_iou_{}'.format(obj_class): np.mean(iou)})
        return np.mean(iou)

    def cal_iou_all_class(self):
        for obj_class in opt.file_names['obj_classes']:
            if opt.eval_mode is None:
                self.cal_iou_one_class(obj_class)
            else:
                self.cal_iou_one_class_one_ins(obj_class)

    def cal_iou_one_class(self, obj_class=None, pred=None, metric='iou'):
        # if obj_class is None:
        #     obj_class = opt.obj_class
        self.image_names = load_names('test_'+obj_class, self.root_dir, obj_class)
        save_iou_file = opt.init_source + '/iou_{}.mat'.format(obj_class)
        if 'shape' in metric:
            save_iou_file = opt.init_source + '/iou_{}_shape.mat'.format(obj_class)
        if 'aabb' in opt.metric:
            save_iou_file = save_iou_file[:-4] + '_aabb.mat'
        iou = []
        print('loading voxels and calculating iou...')
        source = 'prob'
        version = ''
        # gt_dir = self.root_dir + '/ModelNet10_mesh/{}/modelnet_{}_test.mat'.format(obj_class, obj_class)
        # gt = scipy.io.loadmat(gt_dir)['voxTile']
        if opt.encoder in ['resnet', 'hg', 'depth_new']:
            gt_from_prim_dir = os.path.join(self.root_dir, opt.file_names['voxel'])
        else:
            gt_from_prim_dir = self.root_dir + '/ModelNet10_mesh/{}/Myvoxset_{}.mat'.format(obj_class, obj_class)
        gt = scipy.io.loadmat(gt_from_prim_dir)['voxTile']
        if pred is None:
            pred_dir = opt.init_source + '/Myvoxset_prnn_test_{}_lrbound51.mat'.format(obj_class)
            pred = scipy.io.loadmat(pred_dir)['voxTile']
        voxel_length = pred.shape[0]
        assert voxel_length == len(self.image_names)
        print(voxel_length)
        for cnt in tqdm(range(voxel_length)):
            voxel1 = pred[cnt]
            # if source in ['prob', 'torch_test', 'torch2', '3D-PRNN-master']:
            # if opt.rgb_con:
            if opt.encoder in ['resnet', 'hg', 'depth_new']:
                image_name = self.image_names[cnt]
                id_img_ori = int(image_name[0:4])
                cnt_gt = get_model_id(id_img_ori, self.match_id)    # 0-215   # pick out of 216
                voxel2 = gt[cnt_gt]
                # voxel1 = pred[cnt_gt]#rebuttal
            else:
                if source == 'gt':
                    cnt_gt = cnt
                else:
                    cnt_gt = cnt // 5 // (10 // opt.test_gap)
                voxel2 = gt[cnt_gt]
                import ipdb;ipdb.set_trace()##rebuttal, but notice bug if use this
                # voxel1 = pred[cnt_gt]
                if not (obj_class == 'table' and cnt_gt == 16):
                    if not obj_class == 'chair':
                        voxel2 = np.rot90(voxel2, 1, (0, 1))
            # if cnt == 5:  # verify rotation effects
            #     print(cnt//5)
            #     voxel2 = np.rot90(voxel2, 1, (0,1))
            #     scipy.io.savemat('rot.mat', {'rot':voxel2})
            iou_score = np.sum(np.logical_and(voxel1, voxel2)) / np.sum(np.logical_or(voxel1, voxel2))
            if np.isnan(iou_score):
                iou_score = 0.
                print("Warning: found empty voxel pairs at cnt %f" % cnt)
            # else:
            iou.append(iou_score)
        iou = np.array(iou)
        print('iou', obj_class, np.mean(iou), iou.shape)
        scipy.io.savemat(save_iou_file, {'iou': iou[:, np.newaxis],
                                         'mean_iou': np.mean(iou)})
        return np.mean(iou)

    def cal_acc_one_class(self, obj_class=None):
        # if obj_class is None:
        #     obj_class = opt.obj_class
        self.image_names = load_names('test_'+obj_class, self.root_dir, obj_class)
        save_acc_file = opt.init_source + '/acc_{}.mat'.format(obj_class)
        acc = {}
        num = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        gt_dir = self.root_dir + opt.file_names['norm_label']
        gt = torch.load(gt_dir) # list - dict -list
        pred_dir = opt.init_source + '/test_res_mn_{}.mat'.format(obj_class)
        pred = scipy.io.loadmat(pred_dir)['cls']
        pred_length = pred.shape[0]
        print(pred_length)
        for cnt in tqdm(range(pred_length)):
            cls_pred = pred[cnt]
            image_name = self.image_names[cnt]
            id_img_ori = int(image_name[0:4])
            cnt_gt = get_model_id(id_img_ori, self.match_id)    # 0-215   # pick out of 216
            cls_gt = gt[cnt_gt][b'cls'] - 1
            for i in range(len(cls_gt)):
                if i % 3 == 0:
                    num[cls_gt[i]] += 1
        print('acc', obj_class, acc)
        # scipy.io.savemat(save_acc_file, {'iou_{}'.format(obj_class): iou[:, np.newaxis],
        #                                  'mean_iou_{}'.format(obj_class): np.mean(iou)})
        # return np.mean(iou)

    def compute_box2d_iou(self, res, gt, inverse=False):
        # import pdb
        # pdb.set_trace()
        # res = torch.from_numpy(res)
        res_min = res[:2]
        res_max = res[2:]
        res_len = res_max - res_min
        if torch.sum(res_len > 0) < 2:
            return float(-inverse)      # 0 / -1
        gt_min = gt[:2]
        gt_max = gt[2:]
        gt_len = gt_max - gt_min
        if torch.sum(gt_len > 0) < 2:
            # print('error in gt', gt, gt_len)
            return inverse-1        # -1 / 0
        inter_min = torch.max(res_min, gt_min)
        inter_max = torch.min(res_max, gt_max)
        inter_len = inter_max - inter_min
        if torch.sum(inter_len > 0) < 2:
            return 0.
        inter = inter_len[0]*inter_len[1]
        union = res_len[0]*res_len[1] + gt_len[0]*gt_len[1] - inter
        assert inter > 0 and union > 0
        return (inter / union).item()

    def compute_cls_box2d_acc(self, res_cls_i, gt_cls, res_to_gt_match_id, cls_count, box2d_count, box2d_iou,
                              res_box2d_i, gt_box2d, res_ori_num, gt_ori_num, inverse=False):
        for ii in range(len(res_to_gt_match_id)):
            res_gt_ii = res_to_gt_match_id[ii]
            res_cls_tmp = res_cls_i[ii]
            gt_cls_tmp = gt_cls[res_gt_ii]
            cls_count[0][0] += 1    # tp+fn
            cls_count[res_cls_tmp][0] += 1
            cls_count[0][2] += 1    # tp+fp
            cls_count[gt_cls_tmp][2] += 1
            if res_cls_tmp == gt_cls_tmp:   # tp
                cls_count[0][1] += 1
                cls_count[res_cls_tmp][1] += 1

            if ii >= res_ori_num or res_gt_ii >= gt_ori_num:
                continue
            res_box2d_tmp = res_box2d_i[ii]
            gt_box2d_tmp = gt_box2d[res_gt_ii]
            iou_ii = self.compute_box2d_iou(res_box2d_tmp, gt_box2d_tmp, inverse)
            if iou_ii >= 0:
                box2d_iou[0] += iou_ii
                box2d_iou[res_cls_tmp] += iou_ii
                box2d_count[0][0] += 1  # tp+fn
                box2d_count[res_cls_tmp][0] += 1
                box2d_count[0][2] += 1  # tp+fp
                box2d_count[gt_cls_tmp][2] += 1
                if iou_ii > 0.5:    # tp
                    box2d_count[0][1] += 1
                    box2d_count[res_cls_tmp][1] += 1
        return cls_count, box2d_count, box2d_iou

    def replace_res_with_gt(self, res_point, gt_point, match_to_res_id):
        for res_i in range(len(match_to_res_id)):
            gt_i = match_to_res_id[res_i]
            res_point[res_i] = copy.deepcopy(gt_point[gt_i])
        return res_point

    def replace_res_with_gt_prim(self, res_prim, gt_prim):
        # (n, 20)
        st_criterion = torch.nn.L1Loss()
        res_st, gt_st = [], []
        num_res = res_prim.shape[0]
        num_gt = gt_prim.shape[0]
        for i in range(num_res):
            res_st.append(torch.from_numpy(res_prim[i, 10:16]).float())
        for j in range(num_gt):
            gt_st.append(torch.from_numpy(gt_prim[j, 10:16]).float())
        dis = np.zeros((num_res, num_gt))
        for i in range(num_res):
            for j in range(num_gt):
                dis[i, j] = st_criterion(res_st[i], gt_st[j]).item()
        res_id = list(range(num_res))
        gt_id = list(range(num_gt))
        for i in range(num_res):
            dis_min_id = np.argmin(dis)
            res_i_i = dis_min_id // dis.shape[1]
            gt_i_i = dis_min_id % dis.shape[1]
            res_i = res_id.pop(res_i_i)
            gt_i = gt_id.pop(gt_i_i)
            res_prim[res_i] = copy.deepcopy(gt_prim[gt_i])
            dis = np.delete(dis, res_i_i, axis=0)
            dis = np.delete(dis, gt_i_i, axis=1)
            if dis.shape[0] == 0 or dis.shape[1] == 0:
                break
        return res_prim

    def num_match(self, p1, p2, cls_prob_1, cls_prob_2):
        delete_id = []
        length = len(p1)
        dis = np.zeros((length, length))
        for i in range(length):
            for j in range(length):
                if i <= j:
                    dis[i, j] = 1000
                else:
                    # diag_1 = np.linalg.norm(p1[j][6] - p1[j][0])
                    diag_1 = np.linalg.norm(p1[i][6] - p1[i][0])
                    diag_2 = np.linalg.norm(p2[j][6] - p2[j][0])
                    diag = (diag_1 + diag_2) / 2
                    dis[i, j] = directed_hausdorff(p1[i], p2[j])[0] / diag  # 8 points of both
        p1_id = list(range(length))
        p2_id = list(range(length))
        for i in range(length):
            dis_min = np.min(dis)
            if dis_min > opt.nms_thresh:
                break
            dis_min_id = np.argmin(dis)
            p1_i_i = dis_min_id // dis.shape[1]
            p2_i_i = dis_min_id % dis.shape[1]
            p1_i = p1_id[p1_i_i]
            p2_i = p2_id[p2_i_i]
            if max(cls_prob_1[p1_i]) < max(cls_prob_2[p2_i]):
                di_1 = p1_id.pop(p1_i_i)
                dis = np.delete(dis, p1_i_i, axis=0)
                di_2 = p2_id.pop(p1_i_i)
                dis = np.delete(dis, p1_i_i, axis=1)
                delete_id.append(p1_i)
                assert di_1 == di_2
            else:
                di_1 = p1_id.pop(p2_i_i)
                dis = np.delete(dis, p2_i_i, axis=0)
                di_2 = p2_id.pop(p2_i_i)
                dis = np.delete(dis, p2_i_i, axis=1)
                delete_id.append(p2_i)
                assert di_1 == di_2
            if dis.shape[0] == 0 or dis.shape[1] == 0:
                break
        return delete_id

    def num_res(self, res_prim, res_cls_i, res_box2d_i, res_rot_prob_i, res_cls_prob_i, sym_tag):
        res_point = prim_all_to_cornerset(res_prim)
        delete_id = self.num_match(res_point, res_point, res_cls_prob_i, res_cls_prob_i)
        res_prim = np.delete(res_prim, delete_id, axis=0)
        res_cls_i = np.delete(res_cls_i, delete_id, axis=0)
        res_rot_prob_i = np.delete(res_rot_prob_i, delete_id, axis=0)
        res_cls_prob_i = np.delete(res_cls_prob_i, delete_id, axis=0)
        delete_id_box2d = []
        for i in delete_id:
            if sym_tag[i] == 0:
                ori_id = int(np.sum(sym_tag[:i] == 0))
                delete_id_box2d.append(ori_id)  # box2d is half model
        res_box2d_i = np.delete(res_box2d_i, delete_id_box2d, axis=0)
        return res_prim, res_cls_i, res_box2d_i, res_rot_prob_i, res_cls_prob_i, delete_id

    def cal_dist_one_ins(self, i, res, gt, prim_points_2d, model_ids, obj_class, metrics):
        cls_count, box2d_count, box2d_iou, cls_count_pr, box2d_count_pr, box2d_iou_pr, loss_str_all = metrics
        res_null = False
        res_prim, res_cls_i, res_box2d_i, res_rot_prob_i, res_cls_prob_i, sym_tag_i = res_obj_to_gt_prims(i, res)
        num_1 = res_cls_i.shape
        delete_id = []
        if opt.nms_thresh > 0:
            res_prim, res_cls_i, res_box2d_i, res_rot_prob_i, res_cls_prob_i, delete_id = self.num_res(
                res_prim, res_cls_i, res_box2d_i, res_rot_prob_i, res_cls_prob_i, sym_tag_i)
        num_2 = res_cls_i.shape
        res_ori_num = res_box2d_i.shape[0]
        res_box2d_i = torch.from_numpy(res_box2d_i)

        gt_id = model_ids[i]
        gt_prim_ori = gt[gt_id, 0]['ori'][0, 0]
        gt_prim_sym = gt[gt_id, 0]['sym'][0, 0]
        gt_cls_ori = gt[gt_id, 0]['cls'][0, 0][0]
        gt_ori_num = gt_cls_ori.shape[0]
        if np.sum(gt_prim_sym):
            gt_prim_sym, gt_cls_sym = remove_empty_prim_in_primset_sym(gt_prim_sym, gt_cls_ori)
            gt_prim = np.vstack((gt_prim_ori, gt_prim_sym))
            gt_cls = np.hstack((gt_cls_ori, gt_cls_sym))
        else:
            gt_prim = gt_prim_ori
            gt_cls = gt_cls_ori
        image_name = self.image_names[i]
        id_img_ori = int(image_name[0:4])  # 1-3839
        image = load_image(image_name, self.root_dir)
        gt_box2d, _ = get_prim_box_2d(i, self.match_id, id_img_ori, prim_points_2d,
                                      opt.max_len // 3 * 3, image, self.transform, eval=True, inverse=opt.inverse)
        gt_box2d = torch.transpose(torch.from_numpy(gt_box2d).float(), 0, 1)
        
        res_point = prim_all_to_cornerset(res_prim)
        gt_point = prim_all_to_cornerset(gt_prim)
        if res_point == []:
            res_null = True
            dist_i = 1000
            acc_i = ({t: 0 for t in opt.h_thresh}, 0)
            metrics = (cls_count, box2d_count, box2d_iou, cls_count_pr, box2d_count_pr, box2d_iou_pr, loss_str_all)
            return res_null, dist_i, acc_i, metrics

        D1, match_to_gt_id, _, _ = self.hausdorff(gt_point, res_point, False)
        D2, match_to_res_id, correct_num, box_num = self.hausdorff(res_point, gt_point, True)
        
        dist_i = (D1 + D2) / 2
        acc_i = (correct_num, box_num)
        if opt.n_sem == 1 and opt.loss_c is None:
            gt_cls = gt_cls * 0 + 1
        cls_count, box2d_count, box2d_iou = self.compute_cls_box2d_acc(res_cls_i, gt_cls, match_to_res_id,
                                                             cls_count, box2d_count, box2d_iou,
                                                             res_box2d_i, gt_box2d,
                                                             res_ori_num, gt_ori_num)
        cls_count_pr, box2d_count_pr, box2d_iou_pr = self.compute_cls_box2d_acc(gt_cls, res_cls_i, match_to_gt_id,
                                                                      cls_count_pr, box2d_count_pr, box2d_iou_pr,
                                                                      gt_box2d, res_box2d_i,
                                                                      gt_ori_num, res_ori_num,
                                                                      inverse=True)# recall, not precision
        if 'str' in opt.metric:
            losses = self.compute_loss_bidirection(gt_prim, res_prim, match_to_gt_id, match_to_res_id, obj_class,
                                                   res_rot_prob_i, res_cls_prob_i, gt_cls)
            loss_st, loss_r, loss_r_theta, loss_r_axis, loss_c = losses
            loss_str_all[0] += loss_st
            loss_str_all[1] += loss_r
            loss_str_all[2] += loss_r_theta
            loss_str_all[3] += loss_r_axis
            loss_str_all[4] += loss_c
            loss_str_all[5].append(loss_r)
        metrics = (cls_count, box2d_count, box2d_iou, cls_count_pr, box2d_count_pr, box2d_iou_pr, loss_str_all)
        return res_null, dist_i, acc_i, metrics, delete_id

    def cal_dist_one_ins_recall(self, i, res, gt, prim_points_2d, model_ids, obj_class, metrics):
        cls_count, box2d_count, box2d_iou, cls_count_pr, box2d_count_pr, box2d_iou_pr, loss_str_all = metrics
        res_null = False
        if opt.recall_tmp: # 11.5
            res, res_inverse = res
        res_prim, res_cls_i, res_box2d_i, res_rot_prob_i, res_cls_prob_i, sym_tag_i = res_obj_to_gt_prims(i, res)
        num_1 = res_cls_i.shape
        delete_id = []
        if opt.nms_thresh > 0:
            res_prim, res_cls_i, res_box2d_i, res_rot_prob_i, res_cls_prob_i, delete_id = self.num_res(
                res_prim, res_cls_i, res_box2d_i, res_rot_prob_i, res_cls_prob_i, sym_tag_i)
        num_2 = res_cls_i.shape
        if opt.recall_tmp: # 11.5
            res_prim_inv, res_cls_i_inv, res_box2d_i_inv, \
            res_rot_prob_i_inv, res_cls_prob_i_inv, sym_tag_i = res_obj_to_gt_prims(i, res_inverse)
            res_prim = np.vstack((res_prim, res_prim_inv))
            res_cls_i = np.hstack((res_cls_i, res_cls_i_inv))
            res_box2d_i = np.vstack((res_box2d_i, res_box2d_i_inv))
            res_rot_prob_i = np.vstack((res_rot_prob_i, res_rot_prob_i_inv))
            res_cls_prob_i = np.vstack((res_cls_prob_i, res_cls_prob_i_inv))
        res_ori_num = res_box2d_i.shape[0]
        res_box2d_i = torch.from_numpy(res_box2d_i)

        gt_id = model_ids[i]
        gt_prim_ori = gt[gt_id, 0]['ori'][0, 0]
        gt_prim_sym = gt[gt_id, 0]['sym'][0, 0]
        gt_cls_ori = gt[gt_id, 0]['cls'][0, 0][0]
        gt_ori_num = gt_cls_ori.shape[0]
        if np.sum(gt_prim_sym):
            gt_prim_sym, gt_cls_sym = remove_empty_prim_in_primset_sym(gt_prim_sym, gt_cls_ori)
            gt_prim = np.vstack((gt_prim_ori, gt_prim_sym))
            gt_cls = np.hstack((gt_cls_ori, gt_cls_sym))
        else:
            gt_prim = gt_prim_ori
            gt_cls = gt_cls_ori
        res_point = prim_all_to_cornerset(res_prim)
        gt_point = prim_all_to_cornerset(gt_prim)
        if res_point == []:
            res_null = True
            dist_i = 1000
            acc_i = ({t: 0 for t in opt.h_thresh}, 0)
            metrics = (cls_count, box2d_count, box2d_iou, cls_count_pr, box2d_count_pr, box2d_iou_pr, loss_str_all)
            return res_null, dist_i, acc_i, metrics

        D1, match_to_gt_id, correct_num, box_num = self.hausdorff_recall(gt_point, res_point, True)
        D2, match_to_res_id, _, _ = self.hausdorff(res_point, gt_point, False)

        dist_i = (D1 + D2) / 2
        acc_i = (correct_num, box_num)
        metrics = (cls_count, box2d_count, box2d_iou, cls_count_pr, box2d_count_pr, box2d_iou_pr, loss_str_all)
        return res_null, dist_i, acc_i, metrics, delete_id

    def cal_dist_one_class(self, obj_class=None):
        save_dist_file = opt.init_source + '/dist_{}_{}.mat'.format(obj_class, opt.eval_mode)
        if 'aabb' in opt.metric:
            save_dist_file = save_dist_file[:-4] + '_aabb.mat'
        self.image_names = load_names('test_' + obj_class, self.root_dir, obj_class)
        model_ids = match_image_names_to_model_ids('test_' + obj_class, obj_class, self.root_dir)
        res = scipy.io.loadmat(opt.init_source + '/test_res_mn_{}.mat'.format(obj_class))
        gt = scipy.io.loadmat(os.path.join(self.root_dir, opt.file_names['primset']))
        prim_points_2d = load_prim_points_2d(self.root_dir, self.obj_class)
        res_x = res['x']  # 269*4,100
        gt = gt['primset']  # 216,1 struct

        if opt.recall_tmp: # 11.5
            pdb.set_trace()
            dir_inverse = 'p02c01_33_v1_20' # 'p02c01_3_v1_20'
            res_inverse = scipy.io.loadmat(os.path.join(opt.exp_prefix, dir_inverse,
                                                        'test_res_mn_{}.mat'.format(obj_class)))
            res = (res, res_inverse)
        # pdb.set_trace()
        num = int(res_x.shape[0] / 4)
        num_valid = num
        num_null_list = []
        dist = np.zeros([num, 1])
        nms_removed = -np.ones((num, 50))
        acc = [{t: 0 for t in opt.h_thresh}, 0]
        cls_count = [[0, 0, 0] for _ in range(opt.n_sem + 1)]   # tp+fp, tp, not used
        box2d_count = [[0, 0, 0] for _ in range(opt.n_sem + 1)]   # tp
        box2d_iou = [0 for _ in range(opt.n_sem + 1)]   # iou
        cls_count_pr = [[0, 0, 0] for _ in range(opt.n_sem + 1)]  # tp+fp, tp, tp+np
        box2d_count_pr = [[0, 0, 0] for _ in range(opt.n_sem + 1)]  # tp
        box2d_iou_pr = [0 for _ in range(opt.n_sem + 1)]  # iou
        loss_str_all = [0, 0, 0, 0, 0, []]    #[[0, 0] for _ in range(opt.n_sem + 1)]
        # loss_r_all = [0, 0]     #[[0, 0] for _ in range(opt.n_sem + 1)]
        for i in range(num):
            metrics = (cls_count, box2d_count, box2d_iou, cls_count_pr, box2d_count_pr, box2d_iou_pr, loss_str_all)
            if 'recall' in opt.metric:
                res_null, dist_i, acc_i, metrics, delete_id = self.cal_dist_one_ins_recall(i, res, gt, prim_points_2d, model_ids,
                                                                                           obj_class, metrics)
            else:
                res_null, dist_i, acc_i, metrics, delete_id = self.cal_dist_one_ins(i, res, gt, prim_points_2d, model_ids,
                                                                                    obj_class, metrics)
            cls_count, box2d_count, box2d_iou, cls_count_pr, box2d_count_pr, box2d_iou_pr, loss_str_all = metrics
            nms_removed[i, :len(delete_id)] = delete_id
            if res_null:
                num_valid -= 1
                num_null_list.append(i)
            else:
                dist[i, :] = dist_i
                for t in opt.h_thresh:
                    acc[0][t] += acc_i[0][t]
                acc[1] += acc_i[1]
        if num_valid < num - 10:
            dist *= 0
                # pdb.set_trace()

        if num_valid == 0:
            num_valid = 1
        loss_str_all[0] = loss_str_all[0] / num_valid * opt.gmm_weight
        loss_str_all[1] = loss_str_all[1] / num_valid * opt.r_weight
        loss_str_all[2] = loss_str_all[2] / num_valid * opt.r_weight
        loss_str_all[3] = loss_str_all[3] / num_valid * opt.r_weight
        loss_str_all[4] = loss_str_all[4] / num_valid * opt.c_weight


        mean_dist = np.sum(dist) / num_valid
        scipy.io.savemat(save_dist_file, {'dist': dist, 'mean_dist': mean_dist, 'nms_removed': nms_removed})

        # acc = distance.thresh_acc()
        # acc = acc[0] / acc[1]
        for t in opt.h_thresh:
            acc[0][t] = acc[0][t] / acc[1]
        acc = acc[0]
        box2d_acc = []
        box2d_iou_all = []
        for xi in range(len(cls_count)):
            if cls_count[xi][0] == 0:
                cls_count[xi][0] = 1
            if box2d_count[xi][0] == 0:
                box2d_count[xi][0] = 1
            box2d_acc.append(box2d_count[xi][1] / box2d_count[xi][0])
            box2d_iou_all.append(box2d_iou[xi] / box2d_count[xi][0])
        cls_acc = [x[1] / x[0] for x in cls_count]  # precision

        c_pr = [[0, 0] for _ in range(opt.n_sem + 1)]
        b_pr = [[0, 0] for _ in range(opt.n_sem + 1)]
        b_iou = [0 for _ in range(opt.n_sem + 1)]
        for xi in range(len(cls_count_pr)):
            if cls_count_pr[xi][0] == 0:
                cls_count_pr[xi][0] = 1
            if cls_count_pr[xi][2] == 0:
                cls_count_pr[xi][2] = 1
            if box2d_count_pr[xi][0] == 0:
                box2d_count_pr[xi][0] = 1
            if box2d_count_pr[xi][2] == 0:
                box2d_count_pr[xi][2] = 1
            c_pr[xi][0] = cls_count_pr[xi][1] / cls_count_pr[xi][0]
            c_pr[xi][1] = cls_count_pr[xi][1] / cls_count_pr[xi][2]
            b_pr[xi][0] = box2d_count_pr[xi][1] / box2d_count_pr[xi][0]
            b_pr[xi][1] = box2d_count_pr[xi][1] / box2d_count_pr[xi][2]
            b_iou[xi] = box2d_iou_pr[xi] / box2d_count_pr[xi][0]

        print('=' * 30)
        print('can pred, cannot pred: ', num_valid, num - num_valid)
        print('wrong num list: ', num_null_list)
        print('mean_distance', mean_dist)
        print('hausdorff acc', acc)
        print('semantic part acc', cls_acc)
        print('box2d acc', box2d_acc, box2d_iou_all)
        print('box2d pr', c_pr, b_pr, b_iou)
        # print('loss', loss_str_all)
        return mean_dist, acc, cls_acc, box2d_acc, box2d_iou_all, c_pr, b_pr, b_iou, loss_str_all
    
    def cal_dist_one_class_tul(self, obj_class=None):
        # self.count_rotation_gt(self.root_dir)
        save_dist_file = opt.init_source + '/dist_{}_{}.mat'.format(obj_class, opt.eval_mode)
        if 'aabb' in opt.metric:
            save_dist_file = save_dist_file[:-4] + '_aabb.mat'
        self.image_names = load_names('test_' + obj_class, self.root_dir, obj_class)
        model_ids = match_image_names_to_model_ids('test_' + obj_class, obj_class, self.root_dir)
        # res = scipy.io.loadmat(opt.init_source + '/test_res_mn_{}.mat'.format(obj_class))
        gt = scipy.io.loadmat(os.path.join(self.root_dir, opt.file_names['primset']))
        prim_points_2d = load_prim_points_2d(self.root_dir, self.obj_class)
        # res_x = res['x']  # 269*4,100
        gt = gt['primset']  # 216,1 struct
        res = None
        num_all = {'chair': 810, 'table': 558, 'sofa': 368, '3dprnnchair': 500, '3dprnntable': 500, '3dprnnnight_stand': 430} # int(res_x.shape[0] / 4)
        num = num_all[obj_class]
        num_valid = num
        num_null_list = []
        dist = np.zeros([num, 1])
        nms_removed = -np.ones((num, 50))
        acc = [{t: 0 for t in opt.h_thresh}, 0]
        cls_count = [[0, 0, 0] for _ in range(opt.n_sem + 1)]   # tp+fp, tp, not used
        box2d_count = [[0, 0, 0] for _ in range(opt.n_sem + 1)]   # tp
        box2d_iou = [0 for _ in range(opt.n_sem + 1)]   # iou
        cls_count_pr = [[0, 0, 0] for _ in range(opt.n_sem + 1)]  # tp+fp, tp, tp+np
        box2d_count_pr = [[0, 0, 0] for _ in range(opt.n_sem + 1)]  # tp
        box2d_iou_pr = [0 for _ in range(opt.n_sem + 1)]  # iou
        loss_str_all = [0, 0, 0, 0, 0, []]    #[[0, 0] for _ in range(opt.n_sem + 1)]
        # loss_r_all = [0, 0]     #[[0, 0] for _ in range(opt.n_sem + 1)]
        for i in range(num):
            metrics = (cls_count, box2d_count, box2d_iou, cls_count_pr, box2d_count_pr, box2d_iou_pr, loss_str_all)
            if 'recall' in opt.metric:
                res_null, dist_i, acc_i, metrics, delete_id = self.cal_dist_one_ins_recall(i, res, gt, prim_points_2d, model_ids,
                                                                                           obj_class, metrics)
            else:
                res_null, dist_i, acc_i, metrics, delete_id = self.cal_dist_one_ins_tul(i, res, gt, prim_points_2d, model_ids,
                                                                                    obj_class, metrics)
            cls_count, box2d_count, box2d_iou, cls_count_pr, box2d_count_pr, box2d_iou_pr, loss_str_all = metrics
            nms_removed[i, :len(delete_id)] = delete_id
            if res_null:
                num_valid -= 1
                num_null_list.append(i)
            else:
                dist[i, :] = dist_i
                for t in opt.h_thresh:
                    acc[0][t] += acc_i[0][t]
                acc[1] += acc_i[1]
        if num_valid < num - 10:
            dist *= 0
                # pdb.set_trace()

        if num_valid == 0:
            num_valid = 1
        loss_str_all[0] = loss_str_all[0] / num_valid * opt.gmm_weight
        loss_str_all[1] = loss_str_all[1] / num_valid * opt.r_weight
        loss_str_all[2] = loss_str_all[2] / num_valid * opt.r_weight
        loss_str_all[3] = loss_str_all[3] / num_valid * opt.r_weight
        loss_str_all[4] = loss_str_all[4] / num_valid * opt.c_weight


        mean_dist = np.sum(dist) / num_valid
        scipy.io.savemat(save_dist_file, {'dist': dist, 'mean_dist': mean_dist, 'nms_removed': nms_removed})

        # acc = distance.thresh_acc()
        # acc = acc[0] / acc[1]
        for t in opt.h_thresh:
            acc[0][t] = acc[0][t] / acc[1]
        acc = acc[0]
        box2d_acc = []
        box2d_iou_all = []
        for xi in range(len(cls_count)):
            if cls_count[xi][0] == 0:
                cls_count[xi][0] = 1
            if box2d_count[xi][0] == 0:
                box2d_count[xi][0] = 1
            box2d_acc.append(box2d_count[xi][1] / box2d_count[xi][0])
            box2d_iou_all.append(box2d_iou[xi] / box2d_count[xi][0])
        cls_acc = [x[1] / x[0] for x in cls_count]  # precision

        c_pr = [[0, 0] for _ in range(opt.n_sem + 1)]
        b_pr = [[0, 0] for _ in range(opt.n_sem + 1)]
        b_iou = [0 for _ in range(opt.n_sem + 1)]
        for xi in range(len(cls_count_pr)):
            if cls_count_pr[xi][0] == 0:
                cls_count_pr[xi][0] = 1
            if cls_count_pr[xi][2] == 0:
                cls_count_pr[xi][2] = 1
            if box2d_count_pr[xi][0] == 0:
                box2d_count_pr[xi][0] = 1
            if box2d_count_pr[xi][2] == 0:
                box2d_count_pr[xi][2] = 1
            c_pr[xi][0] = cls_count_pr[xi][1] / cls_count_pr[xi][0]
            c_pr[xi][1] = cls_count_pr[xi][1] / cls_count_pr[xi][2]
            b_pr[xi][0] = box2d_count_pr[xi][1] / box2d_count_pr[xi][0]
            b_pr[xi][1] = box2d_count_pr[xi][1] / box2d_count_pr[xi][2]
            b_iou[xi] = box2d_iou_pr[xi] / box2d_count_pr[xi][0]

        print('=' * 30)
        print('can pred, cannot pred: ', num_valid, num - num_valid)
        print('wrong num list: ', num_null_list)
        print('mean_distance', mean_dist)
        print('hausdorff acc', acc)
        print('semantic part acc', cls_acc)
        print('box2d acc', box2d_acc, box2d_iou_all)
        print('box2d pr', c_pr, b_pr, b_iou)
        # print('loss', loss_str_all)
        return mean_dist, acc, cls_acc, box2d_acc, box2d_iou_all, c_pr, b_pr, b_iou, loss_str_all

    def compute_loss_bidirection(self, prim_set_1, prim_set_2, match_to_1, match_to_2, obj_class,
                                 res_rot_prob_i, res_cls_prob_i, gt_cls):
        # file_name = opt.file_names['mean_std'][obj_class]
        # mean_std_init = scipy.io.loadmat(os.path.join(self.root_dir, file_name))
        mean_std_init = opt.mean_std
        gt_rot_class = rot_gt_theta_axis_to_class(prim_set_1)
        prim_set_1 = normalize_prim(prim_set_1, mean_std_init)
        prim_set_2 = normalize_prim(prim_set_2, mean_std_init)
        loss_st_1, loss_r_1, loss_r_theta_1, loss_r_axis_1, loss_c_1 = self.compute_loss_unidirection(
            prim_set_1, prim_set_2, match_to_1, gt_rot_class, res_rot_prob_i, gt_cls, res_cls_prob_i, True)
        loss_st_2, loss_r_2, loss_r_theta_2, loss_r_axis_2, loss_c_2 = self.compute_loss_unidirection(
            prim_set_2, prim_set_1, match_to_2, gt_rot_class, res_rot_prob_i, gt_cls, res_cls_prob_i, False)
        return ((loss_st_1 + loss_st_2) / 2, (loss_r_1 + loss_r_2) / 2,
                (loss_r_theta_1 + loss_r_theta_2) / 2, (loss_r_axis_1 + loss_r_axis_2) / 2,
                (loss_c_1 + loss_c_2) / 2)

    def compute_loss_unidirection(self, prim_set_1, prim_set_2, match_to_1,
                                  gt_rot_class, res_rot_prob_i, gt_cls, res_cls_prob_i, to_gt=True):
        prim_set_1 = torch.from_numpy(prim_set_1).float()
        prim_set_2 = torch.from_numpy(prim_set_2).float()
        res_rot_prob_i = torch.from_numpy(res_rot_prob_i).float()
        res_cls_prob_i = torch.from_numpy(res_cls_prob_i).float()
        gt_cls = torch.from_numpy(gt_cls).long()
        str_criterion = torch.nn.L1Loss()
        if opt.out_r == 'class':
            r_criterion = torch.nn.NLLLoss()
        else:
            r_criterion = torch.nn.L1Loss()
        cls_criterion = torch.nn.NLLLoss()
        loss_st = torch.zeros(1)[0]
        loss_r = torch.zeros(1)[0]
        loss_r_theta = torch.zeros(1)[0]
        loss_r_axis = torch.zeros(1)[0]
        loss_c = torch.zeros(1)[0]
        length = len(match_to_1)
        assert length == prim_set_1.size(0)
        for i in range(length):
            # import pdb;pdb.set_trace()
            prim_1 = prim_set_1[i]
            i2 = match_to_1[i]
            prim_2 = prim_set_2[i2]
            loss_st += str_criterion(prim_1[:6], prim_2[:6])
            if to_gt:
                gt_r = gt_rot_class[i]
                res_r = res_rot_prob_i[i2:i2+1]
                gt_c = gt_cls[i:i+1]
                res_c = res_cls_prob_i[i2:i2+1]
            else:
                gt_r = gt_rot_class[i2]
                res_r = res_rot_prob_i[i:i+1]
                gt_c = gt_cls[i2:i2+1]
                res_c = res_cls_prob_i[i:i+1]
            if opt.out_r == 'class':
                loss_r_axis += r_criterion(torch.log(res_r[:, 27:]), gt_r[1:])
                if gt_r[1:].item() > 0:
                    loss_r_theta += r_criterion(torch.log(res_r[:, :27]), gt_r[:1])
                # loss_r = loss_r_theta + loss_r_axis
            # else:
            loss_r += str_criterion(prim_1[6:], prim_2[6:])
            loss_c += cls_criterion(torch.log(res_c), gt_c - 1)
        # loss_st /= length
        # loss_r /= length
        return loss_st.item(), loss_r.item(), loss_r_theta.item(), loss_r_axis.item(), loss_c.item()

    def hausdorff_recall(self, p1, p2, thresh_acc_flag):
        correct_num = {t: 0 for t in opt.h_thresh}
        D1, match_id, correct_num_t, box_num = 0, 0, 0, 0
        for t in opt.h_thresh:
            D1, match_id, correct_num_t, box_num = self.hausdorff_recall_one(p1, p2, thresh_acc_flag, t)
            correct_num[t] = correct_num_t
        return D1, match_id, correct_num, box_num

    def hausdorff_recall_one(self, p1, p2, thresh_acc_flag, t):
        dis_min_all = []
        match_id = []
        correct_num = 0
        box_num = 0
        p1 = copy.deepcopy(p1)
        p2 = copy.deepcopy(p2)
        for j in range(len(p1)):
            if 'recall' in opt.metric:
                if len(p2) == 0:
                    box_num += 1
                    continue
            dis_tmp = []
            for k in range(len(p2)):
                dis = directed_hausdorff(p1[j], p2[k])[0]  # 8 points of both
                dis_tmp.append(dis)
            dis_min = np.min(dis_tmp)
            dis_min_all.append(dis_min)
            dis_min_id = np.argmin(dis_tmp)
            match_id.append(dis_min_id)
            assert len(dis_tmp) == len(p2)

            if thresh_acc_flag:
                if 'recall' in opt.metric:
                    p_min = p1[j][0]
                    p_max = p1[j][6]
                else:
                    p_min = p2[dis_min_id][0]
                    p_max = p2[dis_min_id][6]
                gt_diag = np.linalg.norm(p_max - p_min)
                score = dis_min / gt_diag
                if score < t:
                    correct_num += 1
                    if 'recall' in opt.metric:#only remove res when matched
                        p2.pop(dis_min_id)
                box_num += 1

        D1 = np.mean(dis_min_all)
        # if opt.test_version == '2':
        # box_num = len(p2)
        return D1, match_id, correct_num, box_num

    def hausdorff(self, p1, p2, thresh_acc_flag):
        dis_min_all = []
        match_id = []
        correct_num = {t: 0 for t in opt.h_thresh}
        box_num = 0
        p1 = copy.deepcopy(p1)
        p2 = copy.deepcopy(p2)
        for j in range(len(p1)):
            # if 'recall' in opt.metric:##incorrect in this function
            #     if len(p2) == 0:
            #         box_num += 1
            #         continue
            dis_tmp = []
            for k in range(len(p2)):
                dis = directed_hausdorff(p1[j], p2[k])[0]  # 8 points of both
                dis_tmp.append(dis)
            dis_min = np.min(dis_tmp)
            dis_min_all.append(dis_min)
            dis_min_id = np.argmin(dis_tmp)
            match_id.append(dis_min_id)

            if thresh_acc_flag:
                # if 'recall' in opt.metric:
                #     p_min = p1[j][0]
                #     p_max = p1[j][6]
                # else:
                p_min = p2[dis_min_id][0]
                p_max = p2[dis_min_id][6]
                gt_diag = np.linalg.norm(p_max - p_min)
                score = dis_min / gt_diag
                for t in opt.h_thresh:
                    if score < t:
                        correct_num[t] += 1
                        # if 'recall' in opt.metric:#only remove res when matched
                        #     p2.pop(dis_min_id)
                box_num += 1

        D1 = np.mean(dis_min_all)
        # if opt.test_version == '2':
        # box_num = len(p2)
        return D1, match_id, correct_num, box_num


class Distance_xy(object):
    def __init__(self):
        global opt
        opt = get_opt()
        self.score_b_list = []
        self.thresh = 0.2
        self.box_num = 0

    def hausdorff(self, p1, p2, thresh_acc_flag):
        dis_min_tmp = []
        match_id = []
        for j in range(len(p1)):
            dis_tmp = []
            for k in range(len(p2)):
                dis = directed_hausdorff(p1[j], p2[k])[0]  # 8 points of both
                # print(k, dis)
                dis_tmp.append(dis)
            dis_min = np.min(dis_tmp)
            dis_min_tmp.append(dis_min)
            dis_min_id = np.argmin(dis_tmp)
            match_id.append(dis_min_id)

            if thresh_acc_flag:
                p_min = p2[dis_min_id][0]
                p_max = p2[dis_min_id][6]
                gt_diag = np.linalg.norm(p_max - p_min)
                score = dis_min / gt_diag
                # print('dis_min, gt_diag, score', score, dis_min, gt_diag)
                if score < self.thresh:
                    score_b = 1
                else:
                    score_b = 0
                self.score_b_list.append(score_b)
                self.box_num += 1

        D1 = np.mean(dis_min_tmp)

        return D1, match_id

    def thresh_acc(self):
        acc = np.sum(list(map(lambda x: x == 1, self.score_b_list))) / self.box_num
        print('*' * 33)
        print('thresh_acc', acc)
        return acc


class Prim2VoxAccel(object):
    def __init__(self):
        global opt
        opt = get_opt()
        self.root_dir = opt.data_dir

    def prim_to_vox_one_class_accel_prims(self, i, obj_class=None, prims=None):
        # if obj_class is None:
        #     obj_class = opt.obj_class
        voxel_scale = opt.v_size
        prim_sum = prims.shape[0]
        voxel_sum = np.zeros([voxel_scale, voxel_scale, voxel_scale])
        voxel_idv = np.zeros([prim_sum, voxel_scale, voxel_scale, voxel_scale])
        for j in range(prim_sum):
            prim_r = prims[j, :]
            voxel = np.zeros([voxel_scale, voxel_scale, voxel_scale])
            voxel = self.prim_to_voxel(voxel, prim_r, voxel_scale)
            voxel_sum += voxel
            voxel_idv[j, :, :, :] = copy.deepcopy(voxel)
        voxel_sum = (voxel_sum > 0).astype(np.float32)
        return voxel_sum, voxel_idv

    def prim_to_vox_one_class_accel_primset(self, i, obj_class=None, primset=None):
        # if obj_class is None:
        #     obj_class = opt.obj_class
        if primset is None:
            primset = scipy.io.loadmat(os.path.join(self.root_dir, opt.file_names['primset']))
            primset = primset['primset']
        voxel_scale = opt.v_size
        vox_ori = primset[i, 0]['ori'][0, 0]
        vox_sym = primset[i, 0]['sym'][0, 0]
        vox_cls = primset[i, 0]['cls'][0, 0] - 1    # start from 0
        # print(vox_cls)
        prim_num = vox_ori.shape[0]
        prim_sum = np.sum(np.sum(vox_sym[:, 10:], axis=1)>0) + prim_num
        voxel_sum = np.zeros([voxel_scale, voxel_scale, voxel_scale])
        voxel_idv = np.zeros([prim_sum, voxel_scale, voxel_scale, voxel_scale])
        prims = np.zeros([prim_sum, 20])
        cls = np.zeros([prim_sum])
        i_idv = 0
        for j in range(prim_num):
            prim_r = vox_ori[j, :]
            voxel = np.zeros([voxel_scale, voxel_scale, voxel_scale])
            voxel = self.prim_to_voxel(voxel, prim_r, voxel_scale)
            voxel_sum += voxel
            voxel_idv[i_idv, :, :, :] = copy.deepcopy(voxel)
            prims[i_idv, :] = copy.deepcopy(prim_r)
            cls[i_idv] = copy.deepcopy(vox_cls[0, j])
            i_idv += 1
            prim_r = vox_sym[j, :]
            if np.sum(np.abs(prim_r[10:])) > 0:
                voxel = np.zeros([voxel_scale, voxel_scale, voxel_scale])
                voxel = self.prim_to_voxel(voxel, prim_r, voxel_scale)
                voxel_sum += voxel
                voxel_idv[i_idv, :, :, :] = copy.deepcopy(voxel)
                prims[i_idv, :] = copy.deepcopy(prim_r)
                cls[i_idv] = copy.deepcopy(vox_cls[0, j])
                i_idv += 1
        # print(prim_sum, i_idv)
        voxel_sum = (voxel_sum > 0).astype(np.float32)
        return voxel_sum, voxel_idv, prims, cls

    def prim_to_vox_one_class_accel_one_ins(self, i, obj_class=None, res=None):
        # if obj_class is None:
        #     obj_class = opt.obj_class
        if res is None:
            res = scipy.io.loadmat(opt.init_source + '/test_res_mn_{}.mat'.format(obj_class))
        res_x = res['x']
        res_prim = res_x[i * 4 + 0:i * 4 + 2, :]
        res_rot = res_x[i * 4 + 2, :]
        res_sym = res_x[i + 3, :]
        res_cls = res['cls'][i, :]  # start from 0
        stop_idx = [ind for (ind, x) in enumerate(res_prim[0, :]) if x == 0][0]
        voxel_scale = opt.v_size
        voxel_sum = np.zeros([voxel_scale, voxel_scale, voxel_scale])
        voxel_idv = np.zeros([stop_idx // 3 * 2, voxel_scale, voxel_scale, voxel_scale])
        prims = np.zeros([stop_idx // 3 * 2, 20])
        cls = np.zeros([stop_idx // 3 * 2])
        i_idv = 0
        for col in range(0, stop_idx - 2, 3):
            prim_rot = res_rot[col:col + 3]
            prim_r = np.concatenate((res_prim[0, col:col + 3], res_prim[1, col:col + 3] + opt.shift_res, prim_rot),
                                    axis=0)
            sym_r = res_sym[col:col + 3]
            Rv = prim_r[6:9]
            Rv_y = np.max(np.abs(Rv))
            Rv_i = np.argmax(np.abs(Rv))
            theta = Rv[Rv_i]
            Rv = np.zeros([3])
            Rv[Rv_i] = 1
            prim_r = np.concatenate((np.zeros([10]), prim_r[0:6], Rv, [theta]), axis=0)
            voxel = np.zeros([voxel_scale, voxel_scale, voxel_scale])
            voxel = self.prim_to_voxel(voxel, prim_r, voxel_scale)
            voxel_sum += voxel
            voxel_idv[i_idv, :, :, :] = copy.deepcopy(voxel)
            prims[i_idv, :] = copy.deepcopy(prim_r)
            cls[i_idv] = copy.deepcopy(res_cls[col])
            i_idv += 1
            if prim_r[13] + prim_r[10] >= voxel_scale / 2 or opt.full_model or opt.pqnet:
                continue
            else:
                prim_r = get_sym(prim_r, voxel_scale)
                voxel = np.zeros([voxel_scale, voxel_scale, voxel_scale])
                voxel = self.prim_to_voxel(voxel, prim_r, voxel_scale)
                voxel_sum += voxel
                voxel_idv[i_idv, :, :, :] = copy.deepcopy(voxel)
                prims[i_idv, :] = copy.deepcopy(prim_r)
                cls[i_idv] = copy.deepcopy(res_cls[col])
                i_idv += 1
        voxel_idv = voxel_idv[:i_idv, :, :, :]
        prims = prims[:i_idv, :]
        cls = cls[:i_idv]
        # print(stop_idx // 3 * 2, i_idv)
        # print(voxel_idv.shape, i_idv)
        voxel_sum = (voxel_sum > 0).astype(np.float32)
        return voxel_sum, voxel_idv, prims, cls

    def prim_to_vox_one_class_accel_by_vox(self, obj_class=None):
        t7_dir = os.path.join(self.root_dir, 'gt_pth')
        # if obj_class is None:
        #     obj_class = opt.obj_class
        # since = time.time()
        voxel_scale = opt.v_size
        sample_grid = 7
        # save_file = opt.init_source + '/Myvoxset_prnn_test_{}_lrbound51.mat'.format(obj_class)
        # res = scipy.io.loadmat('../data/3dprnn/sample_generation/test_res_mn_{}_prob.mat'.format(obj_class))
        if opt.save_gt_t7:
            res = scipy.io.loadmat(t7_dir + '/test_res_mn_{}.mat'.format(obj_class))
            gap = 1
        else:
            res = scipy.io.loadmat(opt.init_source + '/test_res_mn_{}.mat'.format(obj_class))
            gap = opt.test_gap
        res = res['x']
        num = int(len(res)/4)
        # voxTile = np.zeros([math.ceil(num/gap), voxel_scale, voxel_scale, voxel_scale])

        for i in range(0, num, gap):
            voxel = np.zeros([voxel_scale, voxel_scale, voxel_scale])
            res_prim = res[i * 4 + 0:i * 4 + 2, :]
            res_rot = res[i * 4 + 2, :]
            res_sym = res[i + 3, :]
            stop_idx = [ind for (ind, x) in enumerate(res_prim[0, :]) if x == 0][0]
            for col in range(0, stop_idx - 2, 3):
                prim_rot = res_rot[col:col + 3]
                prim_r = np.concatenate((res_prim[0, col:col + 3], res_prim[1, col:col + 3] + opt.shift_res, prim_rot), axis=0)
                sym_r = res_sym[col:col + 3]
                Rv = prim_r[6:9]
                Rv_y = np.max(np.abs(Rv))
                Rv_i = np.argmax(np.abs(Rv))
                theta = Rv[Rv_i]
                Rv = np.zeros([3])
                Rv[Rv_i] = 1
                prim_r = np.concatenate((np.zeros([10]), prim_r[0:6], Rv, [theta]), axis=0)
                voxel = self.prim_to_voxel(voxel, prim_r, voxel_scale)
                if prim_r[13] + prim_r[10] >= voxel_scale / 2 or opt.full_model or opt.pqnet:
                    continue
                else:
                    prim_r = get_sym(prim_r, voxel_scale)
                    voxel = self.prim_to_voxel(voxel, prim_r, voxel_scale)
            save_file = 'voxel{}.mat'.format(i)
            if opt.save_gt_t7:
                save_dir = t7_dir + '/vox'
            else:
                save_dir = opt.init_source + '/vox'
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_dir = os.path.join(save_dir, obj_class)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            scipy.io.savemat(os.path.join(save_dir, save_file), {'voxTile': voxel})
            if np.mod(i, 100*opt.test_gap) == 0:
                print(i)
        #     voxTile[int(np.ceil(i / gap)), :, :, :] = voxel
        #     if np.mod(i, 1000) == 0:
        #         print(i)
        #         scipy.io.savemat(save_file, {'voxTile': voxTile})
        #     # if i == 0 :
        #     #     io.savemat(save_file, {'voxTile':voxTile})
        #     #     break
        #     #print(time.time() - since) # spend time
        # scipy.io.savemat(save_file, {'voxTile': voxTile})

    def prim_to_vox_all_class(self):
        for obj_class in opt.file_names['obj_classes']:
            self.prim_to_vox_one_class_accel(obj_class)

    def prim_to_vox_one_class_accel(self, obj_class=None, nms_removed=None, aabb=False):
        # if obj_class is None:
        #     obj_class = opt.obj_class
        # since = time.time()
        nms_removed_i = []
        voxel_scale = opt.v_size
        sample_grid = 7
        save_file = opt.init_source + '/Myvoxset_prnn_test_{}_lrbound51.mat'.format(obj_class)
        # res = scipy.io.loadmat('../data/3dprnn/sample_generation/test_res_mn_{}_prob.mat'.format(obj_class))
        res_all = scipy.io.loadmat(opt.init_source + '/test_res_mn_{}.mat'.format(obj_class))
        res = res_all['x']
        prim_cls = res_all['cls']
        gap = opt.test_gap
        num = int(len(res)/4)
        voxTile = np.zeros([math.ceil(num/gap), voxel_scale, voxel_scale, voxel_scale])

        for i in range(0, num, gap):
            if opt.nms_thresh > 0:
                nms_removed_i = nms_removed[i].tolist()
            voxel_sum = np.zeros([voxel_scale, voxel_scale, voxel_scale])
            res_prim = res[i * 4 + 0:i * 4 + 2, :]
            res_rot = res[i * 4 + 2, :]
            res_sym = res[i + 3, :]
            cls_i = prim_cls[i]
            # print(cls_i)
            # print(np.unique(cls_i))
            stop_idx = [ind for (ind, x) in enumerate(res_prim[0, :]) if x == 0][0]
            count = 0
            for col in range(0, stop_idx - 2, 3):
                prim_rot = res_rot[col:col + 3]
                prim_r = np.concatenate((res_prim[0, col:col + 3], res_prim[1, col:col + 3] + opt.shift_res, prim_rot), axis=0)
                sym_r = res_sym[col:col + 3]
                cls_r = cls_i[col]
                Rv = prim_r[6:9]
                Rv_y = np.max(np.abs(Rv))
                Rv_i = np.argmax(np.abs(Rv))
                theta = Rv[Rv_i]
                Rv = np.zeros([3])
                Rv[Rv_i] = 1
                prim_r = np.concatenate((np.zeros([10]), prim_r[0:6], Rv, [theta]), axis=0)
                if not (opt.nms_thresh > 0 and count in nms_removed_i) and cls_r < opt.n_sem:   # nyu no cls
                    voxel = np.zeros([voxel_scale, voxel_scale, voxel_scale])
                    voxel = self.prim_to_voxel(voxel, prim_r, voxel_scale, aabb)
                    voxel_sum += voxel
                count += 1
                if prim_r[13] + prim_r[10] >= voxel_scale / 2 or opt.full_model or opt.pqnet:
                    continue
                else:
                    if not (opt.nms_thresh > 0 and count in nms_removed_i) and cls_r < opt.n_sem:
                        prim_r = get_sym(prim_r, voxel_scale)
                        voxel = np.zeros([voxel_scale, voxel_scale, voxel_scale])
                        voxel = self.prim_to_voxel(voxel, prim_r, voxel_scale, aabb)
                        voxel_sum += voxel
                    count += 1
            voxel_sum = (voxel_sum > 0).astype(np.float32)
            voxTile[int(np.ceil(i / gap)), :, :, :] = voxel_sum
            if np.mod(i, 100*opt.test_gap) == 0:
                print(i)
                # scipy.io.savemat(save_file, {'voxTile': voxTile})
            # if i == 0 :
            #     io.savemat(save_file, {'voxTile':voxTile})
            #     break
            #print(time.time() - since) # spend time
            # import ipdb;ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        # scipy.io.savemat(save_file, {'voxTile': voxTile})
        return voxTile

    def check_voxel_in_box(self, voxel, voxel_coor, shape, trans, Rv, theta, cen_mean,voxel_scale):
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
        # print(cnt)
        return voxel

    def prim_to_voxel(self, voxel, prim_r, voxel_scale, aabb=False):
        cnt = 0
        voxel_one = np.zeros([27001,1])
        shape = prim_r[10:13]
        trans = prim_r[13:16]
        Rv = prim_r[16:19]
        Rv = np.array(Rv,dtype = int)
        theta = prim_r[19]
        scale = [1,1,1]
        vx = getVx(Rv)
        #Rrot = math.cos(theta)*np.eye(3) + math.sin(theta)*vx + (1-math.cos(theta)) * Rv.T * Rv

        Rrot = math.cos(theta)*np.eye(3) + math.sin(theta)*vx + (1-math.cos(theta)) * np.array([Rv],).T * np.array([Rv],)
        cen_mean = get_mean(shape, trans, scale, Rrot)
        #####meshgrid
        vx = np.arange(voxel_scale)
        vy = np.arange(voxel_scale)
        vz = np.arange(voxel_scale)
        vx, vy, vz = np.meshgrid(vx, vy, vz)
        vx = np.reshape(vx, -1)
        vy = np.reshape(vy, -1)
        vz = np.reshape(vz, -1)
        voxel_coor = np.vstack((vy, vx, vz)).T + 1
        # print(np.all(voxel_coor==voxel_coor1))
        # import pdb;pdb.set_trace()
        voxel = self.check_voxel_in_box(voxel, voxel_coor,shape,trans, Rv,theta,cen_mean,voxel_scale)
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


class Prim2Vox(object):
    def __init__(self):
        global opt
        opt = get_opt()

    def prim_to_vox_all_class(self):
        for obj_class in opt.file_names['obj_classes']:
            self.prim_to_vox_one_class(obj_class)

    def prim_to_vox_one_class(self, obj_class=None):
        # if obj_class is None:
        #     obj_class = opt.obj_class
        voxel_scale = opt.v_size
        sample_grid = 7
        save_file = opt.init_source + '/Myvoxset_prnn_test_{}_lrbound51.mat'.format(obj_class)
        # res = scipy.io.loadmat('../data/3dprnn/sample_generation/test_res_mn_{}_prob.mat'.format(obj_class))
        res = scipy.io.loadmat(opt.init_source + '/test_res_mn_{}.mat'.format(obj_class))
        res = res['x']
        gap = opt.test_gap
        num = int(len(res)/4)
        voxTile = np.zeros([math.ceil(num/gap), voxel_scale, voxel_scale, voxel_scale])

        for i in range(0, num, gap):
            voxel = np.zeros([voxel_scale, voxel_scale, voxel_scale])
            res_prim = res[i * 4 + 0:i * 4 + 2, :]
            res_rot = res[i * 4 + 2, :]
            res_sym = res[i + 3, :]
            stop_idx = [ind for (ind, x) in enumerate(res_prim[0, :]) if x == 0][0]
            for col in range(0, stop_idx - 2, 3):
                prim_rot = res_rot[col:col + 3]
                prim_r = np.concatenate((res_prim[0, col:col + 3], res_prim[1, col:col + 3] + opt.shift_res, prim_rot), axis=0)
                sym_r = res_sym[col:col + 3]
                Rv = prim_r[6:9]
                Rv_y = np.max(np.abs(Rv))
                Rv_i = np.argmax(np.abs(Rv))
                theta = Rv[Rv_i]
                Rv = np.zeros([3])
                Rv[Rv_i] = 1
                prim_r = np.concatenate((np.zeros([10]), prim_r[0:6], Rv, [theta]), axis=0)
                voxel = self.prim_to_voxel(voxel, prim_r, voxel_scale)
                if prim_r[13] + prim_r[10] >= voxel_scale / 2 or opt.full_model or opt.pqnet:
                    continue
                else:
                    prim_r = get_sym(prim_r, voxel_scale)
                    voxel = self.prim_to_voxel(voxel, prim_r, voxel_scale)
            # import pdb
            # pdb.set_trace()
            voxTile[int(np.ceil(i / gap)), :, :, :] = voxel
            if np.mod(i, 100*opt.test_gap) == 0:
                print(i)
                scipy.io.savemat(save_file, {'voxTile': voxTile})
            # if i == 30 :
            #     io.savemat(save_file, {'voxTile':voxTile})
        scipy.io.savemat(save_file, {'voxTile': voxTile})

    def check_voxel_in_box(self, voxel, shape, trans, Rv, theta, cen_mean):
        # voxel = np.array([1,1,1])
        # cen_mean = np.array([14.343102,22.324961,23.012661])
        in_box = 0
        voxel = voxel - 0.5
        voxel = voxel - cen_mean
        theta = -theta
        vx = getVx(Rv)
        Rrot = math.cos(theta) * np.eye(3) + math.sin(theta) * vx + (1 - math.cos(theta)) * np.array([Rv],).T * np.array([Rv],)
        voxel = voxel.dot(Rrot)
        voxel = voxel + cen_mean
        voxel = voxel - trans
        thresh = 0
        lbound = 0.5
        rbound = 0.1
        if (voxel[0] >= -lbound or np.abs(voxel[0]) < thresh) and (
                voxel[0] <= (shape[0] - rbound) or np.abs(voxel[0] - shape[0]) < thresh):
            if (voxel[1] >= -lbound or np.abs(voxel[1]) < thresh) and (
                    voxel[1] <= (shape[1] - rbound) or np.abs(voxel[1] - shape[1]) < thresh):
                if (voxel[2] >= -lbound or np.abs(voxel[2]) < thresh) and (
                        voxel[2] <= (shape[2] - rbound) or np.abs(voxel[2] - shape[2]) < thresh):
                    in_box = 1
        return in_box

    def prim_to_voxel(self, voxel, prim_r, voxel_scale):
        cnt = 0
        voxel_one = np.zeros([27001,1])
        shape = prim_r[10:13]
        trans = prim_r[13:16]
        Rv = prim_r[16:19]
        Rv = np.array(Rv,dtype = int)
        theta = prim_r[19]
        scale = [1,1,1]
        vx = getVx(Rv)
        #Rrot = math.cos(theta)*np.eye(3) + math.sin(theta)*vx + (1-math.cos(theta)) * Rv.T * Rv

        Rrot = math.cos(theta)*np.eye(3) + math.sin(theta)*vx + (1-math.cos(theta)) * np.array([Rv],).T * np.array([Rv],)
        cen_mean = get_mean(shape, trans, scale, Rrot)

        for vx in range(1,voxel_scale+1):
            for vy in range(1,voxel_scale+1):
                for vz in range(1,voxel_scale+1):
                    point = np.array([vx,vy,vz])
                    in_box = self.check_voxel_in_box(point,shape,trans,Rv,theta,cen_mean)
                    if in_box:
                        voxel[vx-1,vy-1,vz-1]=1
                        #print(voxel[vx,vy,vz])
                        voxel_one[cnt, 0] = 1
                        cnt += 1

        # print(cnt)
        return voxel


def evaluate_3dbbox(outputs, parts, existence):
    global opt
    opt = get_opt()
    if opt.exist:
        out_parts, out_exist = outputs
        acc = compute_acc_3dbbox(out_exist, existence)
        iou = compute_miou_3dbbox(out_parts, parts, existence)
    else:
        out_parts = outputs
        # acc = Variable(-torch.ones(1)[0])
        acc = {x: -torch.ones(1)[0] for x in list(range(opt.n_sem)) + ['mean']}
        if opt.use_gpu:
            acc = {x: acc[x].cuda() for x in list(range(opt.n_sem)) + ['mean']}
        iou = compute_miou_3dbbox(out_parts, parts, existence)

    return iou, acc


def compute_acc_3dbbox(out_exist, existence):
    # out_exist size([batch, 6])
    # acc = (((out_exist >= 0.) == existence.byte()).float()).mean()
    acc = {}
    # mean_acc = Variable(torch.zeros(1)[0])
    mean_acc = torch.zeros(1)[0]
    if opt.use_gpu:
        mean_acc = mean_acc.cuda()
    for i_sem in range(opt.n_sem):
        acc[i_sem] = (((out_exist[:, i_sem] >= 0.) == existence[:, i_sem].byte()).float()).mean()
        mean_acc += acc[i_sem]
    acc['mean'] = mean_acc / opt.n_sem

    return acc


def compute_miou_3dbbox(out_parts, parts, existence):
    # out_parts size([batch, 36])
    # 'mean': tensor(0.6535, device='cuda:0')}
    iou = {}
    # mean_iou = Variable(torch.zeros(1)[0])
    mean_iou = torch.zeros(1)[0]
    if opt.use_gpu:
        mean_iou = mean_iou.cuda()
    bad_prim_count = 0.
    for i_sem in range(opt.n_sem):
        min_out = out_parts[:, i_sem * opt.n_para + 0: i_sem * opt.n_para + 3]
        len_out = out_parts[:, i_sem * opt.n_para + 3: i_sem * opt.n_para + 6]
        max_out = min_out + len_out
        min_lab = parts[:, i_sem * opt.n_para + 0: i_sem * opt.n_para + 3]
        len_lab = parts[:, i_sem * opt.n_para + 3: i_sem * opt.n_para + 6]
        max_lab = min_lab + len_lab
        inter_min = torch.max(min_out, min_lab)
        inter_max = torch.min(max_out, max_lab)
        inter_len = inter_max - inter_min
        inter = inter_len[:, 0] * inter_len[:, 1] * inter_len[:, 2]
        union = len_out[:, 0] * len_out[:, 1] * len_out[:, 2]
        union += len_lab[:, 0] * len_lab[:, 1] * len_lab[:, 2]
        union -= inter
        exist_mask = existence[:, i_sem]
        inter_mask = ((inter_len[:, 0] > 0.) * (inter_len[:, 1] > 0.) * (inter_len[:, 2] > 0.)).float()
        inter *= exist_mask
        union *= exist_mask
        i_iou = inter / (union + 1. - exist_mask)   # prevent nan
        i_iou *= inter_mask
        if torch.sum(exist_mask) > 0.:
            iou[i_sem] = i_iou.sum() / torch.sum(exist_mask)
            mean_iou += iou[i_sem]
        else:
            # iou[i_sem] = Variable(-torch.ones(1)[0])
            iou[i_sem] = -torch.ones(1)[0]
            if opt.use_gpu:
                iou[i_sem] = iou[i_sem].cuda()
            bad_prim_count += 1.
    if bad_prim_count == opt.n_sem:
        # iou['mean'] = Variable(torch.ones(1)[0])
        iou['mean'] = torch.ones(1)[0]
        if opt.use_gpu:
            iou['mean'] = iou['mean'].cuda()
    else:
        iou['mean'] = mean_iou / (opt.n_sem - bad_prim_count)

    return iou
