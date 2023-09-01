import torch
import torch.nn as nn
import collections
import copy
import math
import scipy.io
import ipdb
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.structures.bounding_box import BoxList

from lib.opts import *
from lib.utils.prnn_utils import *
from .bbox_model import define_resnet_encoder
from .graph_model import GATPrim, GraphNet, PrimMLPs


class Faster3DProposal(nn.Module):
    def __init__(self):
        super(Faster3DProposal, self).__init__()
        global opt
        opt = get_opt()
        self.n_sem = opt.n_sem
        if opt.loss_y in ['l2', 'l2c', 'l1', 'l1c']:
            if opt.reg == 'str':
                out_size = 2 * opt.xyz + opt.dim_e + 1 * opt.xyz
            elif opt.reg == 's-t-r':
                out_size = 1 * opt.xyz + opt.dim_e
            else:
                out_size = 2 * opt.xyz + opt.dim_e
            if opt.sem_reg:
                self.y_linear = nn.ModuleList([nn.Sequential(
                    nn.Linear(opt.hid_size * opt.hid_layer, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 64),  # in_size
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, out_size)
                ) for i in range(self.n_sem)])
            else:
                self.y_linear = nn.Sequential(
                    nn.Linear(opt.hid_size * opt.hid_layer, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 64), # in_size
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, out_size)
                )
        else:
            raise NotImplementedError
        if opt.out_r == 'theta':
            if opt.sem_reg:
                self.rot_linear = nn.ModuleList([nn.Sequential(
                    # nn.Linear(opt.hid_size * opt.hid_layer, 1 * opt.xyz)
                    nn.Linear(opt.hid_size * opt.hid_layer, 256),
                    nn.ReLU(inplace=True),
                    # nn.Linear(256, 64),  # in_size
                    # nn.ReLU(inplace=True),
                    # nn.Linear(64, 32),
                    # nn.ReLU(inplace=True),
                    nn.Linear(256, 1 * opt.xyz)
                ) for i in range(self.n_sem)])
            else:
                self.rot_linear = nn.Sequential(
                    # nn.Linear(opt.hid_size * opt.hid_layer, 1 * opt.xyz)
                    nn.Linear(opt.hid_size * opt.hid_layer, 256),
                    nn.ReLU(inplace=True),
                    # nn.Linear(256, 64),  # in_size
                    # nn.ReLU(inplace=True),
                    # nn.Linear(64, 32),
                    # nn.ReLU(inplace=True),
                    nn.Linear(256, 1 * opt.xyz)
                )
        else:
            raise NotImplementedError
        if opt.loss_c is not None:
            self.cls_linear = nn.Sequential(
                nn.Linear(opt.hid_size * opt.hid_layer, self.n_sem),
                nn.Softmax(dim=1)
            )
        self.rgb_encoder = define_resnet_encoder(num_reg=opt.con_size)
        con_size = opt.con_size
        if opt.fpn:
            # scales_all = [1/4, 1/8, 1/16, 1/32]
            scales_all = [1/4]
            self.pooler = Pooler(output_size=(7, 7), scales=scales_all, sampling_ratio=2)
            in_size = 256
            if 'dense' in opt.box2d_en:
                in_size += 2
            self.conv_local = nn.Conv2d(in_size, abs(opt.box2d_size), kernel_size=7, stride=1, padding=0)
            con_size += opt.box2d_size
        self.embed_layer = nn.Sequential(
            nn.Linear(con_size, opt.hid_size * opt.hid_layer),
            nn.ReLU(inplace=True),
            nn.Linear(opt.hid_size * opt.hid_layer, opt.hid_size * opt.hid_layer),
            nn.ReLU(inplace=True)
        )
        if opt.refine == 'graph':
            self.refine_module = nn.ModuleList([GraphNet(opt.node, opt.embed_type)] +
                                               [GraphNet(opt.refine_node, opt.refine_embed)
                                                for _ in range(opt.refine_mp - 1)])
            if opt.sem_final:
                self.prim_mlps = PrimMLPs()

    def get_sequence_len_all(self, input_mat):
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

    def compute_max_len_prop(self, box_gt_idx):
        box_gt_idx_len = box_gt_idx.size(2)
        for i in range(box_gt_idx_len, 0, -1):
            if torch.sum(box_gt_idx[:, :, i - 1]) > -box_gt_idx.size(0):
                return i
        return 0

    def compute_max_len(self, input_mat):
        input_max_len = input_mat.size()[2]
        for i in range(input_max_len, 0, -1):
            if torch.sum(torch.abs(input_mat[:, :, i - 1])) != 0:
                return i
        return 0

    def get_target(self, input_mat, rot_mat, cls_mat, prim_box_2d):
        targets = []
        input_max_len = self.compute_max_len(input_mat)
        for i in range(input_max_len):
            x_target = copy.deepcopy(input_mat[:, :, i])
            r_target = copy.deepcopy(rot_mat[:, :, i])
            c_target = copy.deepcopy(cls_mat[:, 0, i])
            b_target = copy.deepcopy(prim_box_2d[:, :, i])
            targets.append((x_target, r_target, c_target, b_target))
        return targets

    def get_target_matched(self, input_mat, rot_mat, cls_mat, prim_box_2d, num_prop, box_gt_idx):
        batch_size = input_mat.size(0)
        targets = []
        c_mask_mat_new = torch.zeros(batch_size, 1, opt.n_prop).cuda()
        for i in range(num_prop):
            x_target, r_target, c_target, b_target = [], [], [], []
            for b in range(batch_size):
                idx = box_gt_idx[b, 0, i].item()
                if idx >= 0:
                    x_target.append(copy.deepcopy(input_mat[b:b + 1, :, idx]))
                    r_target.append(copy.deepcopy(rot_mat[b:b + 1, :, idx]))
                    c_target.append(copy.deepcopy(cls_mat[b:b + 1, 0, idx]))
                    b_target.append(copy.deepcopy(prim_box_2d[b:b + 1, :, idx]))
                    c_mask_mat_new[b, 0, i] = 1
                else:
                    x_target.append(copy.deepcopy(input_mat[b:b + 1, :, idx])*0)
                    r_target.append(copy.deepcopy(rot_mat[b:b + 1, :, idx])*0)
                    c_target.append(copy.deepcopy(cls_mat[b:b + 1, 0, idx])*0)
                    b_target.append(copy.deepcopy(prim_box_2d[b:b + 1, :, idx])*0)
            x_target = torch.cat(x_target, dim=0)
            r_target = torch.cat(r_target, dim=0)
            c_target = torch.cat(c_target, dim=0)
            b_target = torch.cat(b_target, dim=0)
            targets.append((x_target, r_target, c_target, b_target))
        # c_mask_mat_new = (box_gt_idx >= 0).float()
        return targets, c_mask_mat_new

    def forward(self, input_mat, rot_mat, cls_mat, depth, c_mask_mat, prim_box_2d, box_proposal, box_gt_idx, phase):
        outputs, embeddings = [], []
        # num_prop = box_proposal.size(-1)  # prim_box_2d (16, 4, 33)
        num_prop = self.compute_max_len_prop(box_gt_idx)
        targets, c_mask_mat_new = self.get_target_matched(input_mat, rot_mat, cls_mat,
                                                      prim_box_2d, num_prop, box_gt_idx)
        d = self.rgb_encoder(depth)
        if opt.reg_init != 'None':
            d, _ = d
        if opt.fpn:
            global_encoding, local_features = d # global_encoding (16, 256), local_features p2 map
            local_encoding = self.get_local_feature(local_features, box_proposal[:, :, :num_prop])
            # local_encoding = local_encoding[:, :, i]  # 16, 32
            batch_size, con_size = global_encoding.size()
            global_encoding = global_encoding.unsqueeze(dim=2).expand(batch_size, con_size, num_prop)# 16,256->16,256,33
            d = torch.cat((global_encoding, local_encoding), dim=1) # 16, 256+32, 33
        for i in range(num_prop):
            embedding = self.embed_layer(d[:, :, i])
            # cls_gt = self.get_cls_gt(cls_mat, box_gt_idx[:, 0, i])
            cls_gt = targets[i][2]
            y, rot, cls = self.forward_one_step(embedding, cls_gt=cls_gt, phase=phase)
            box2d = box_proposal[:, :, i]
            outputs.append((y, rot, cls, box2d))
            embeddings.append(embedding)
        if opt.refine is not None:
            length = c_mask_mat_new.sum(dim=2)
            targets = self.get_target(input_mat, rot_mat, cls_mat, prim_box_2d)
            outputs = self.refine_train(length, None, c_mask_mat, outputs, embeddings, targets)
        if opt.demo is not None:
            if opt.refine == 'graph':
                outputs = outputs[-1]
            outputs = [(out[0].cpu().detach(), out[1].cpu().detach(), out[2].cpu().detach(), out[3].cpu().detach())
                       for out in outputs]
        return outputs, (targets, c_mask_mat_new)

    def refine_train(self, length, features, c_mask_mat, outputs, prev_h, targets):
        assert opt.loss_e is None  # otherwise y and x are not the same
        if opt.demo is None:
            phase = 'train'
        else:
            phase = 'val'
            assert opt.reg_init != 'None'
        outputs_refine = []
        if opt.refine == 'graph':
            lengths = length[:, 0].int().tolist()
            nodes = prev_h
            for i in range(opt.refine_mp):
                outputs, nodes = self.refine_module[i](lengths, outputs, prev_h=nodes,
                                                       features=features, targets=targets, phase=phase)
                if opt.out_refine != 'None':
                    outputs = self.refine_output(lengths, outputs, prev_h=prev_h, features=features)
                outputs_refine.append(outputs)
            if opt.sem_final:
                if opt.demo is None:
                    cls_gt_all, res_gt_id_all = get_matched_cls_gt(outputs, targets, c_mask_mat, length)
                    outputs = self.prim_mlps(lengths, outputs, nodes, cls_gt_all, phase=phase)
                    outputs_refine.append((outputs, res_gt_id_all))
                else:
                    outputs = self.prim_mlps(lengths, outputs, nodes, None, phase=phase)
                    outputs_refine.append(outputs)
        return outputs_refine

    def get_cls_gt(self, cls_mat, box_gt_idx_i):
        cls_gt = []
        for b in range(box_gt_idx_i.size(0)):
            idx = box_gt_idx_i[b].item()
            if idx >= 0:
                cls_gt.append(cls_mat[b, 0, idx].item())
            else:
                cls_gt.append(-1)
        return torch.Tensor(cls_gt).cuda()

    def forward_one_step(self, h, cls=None, cls_gt=None, phase=None):
        cls_update = cls
        if opt.loss_c is not None:
            cls_update = self.cls_linear(h)
            if opt.sem_select == 'gt' and phase == 'train' and opt.refine == None and opt.bi_lstm == None:
                cls = cls_gt
            else:
                cls = cls_update
        if opt.sem_reg:
            y = self.forward_by_ins(h, cls, out='y', phase=phase)
        else:
            y = self.y_hat(self.y_linear(h))
        if opt.out_r == 'theta':
            if opt.sem_reg:
                rot = self.forward_by_ins(h, cls, out='rot', phase=phase)
            else:
                rot = self.rot_linear(h)
        else:
            raise NotImplementedError
        return y, rot, cls_update

    def forward_by_ins(self, h, cls, out=None, phase=None):
        if h.size(0) <= self.n_sem:
            return self.forward_by_loop(h, cls, out, phase)
        else:
            return self.forward_by_mask(h, cls, out, phase)

    def forward_by_loop(self, h, cls, out=None, phase=None):
        y = []
        for ins_i in range(cls.size(0)):
            if opt.sem_select == 'gt' and phase == 'train' and opt.refine == None and opt.bi_lstm == None:
                cls_id = cls[ins_i].item()
            else:
                _, cls_id = torch.max(cls[ins_i, :], 0)
                cls_id = cls_id.item()
            if out == 'y':
                if opt.reg is not None:
                    y.append(self.y_linear[cls_id](h[ins_i:ins_i + 1]))
                else:
                    y.append(self.y_linear[cls_id](h[ins_i:ins_i + 1]))
            elif out == 'rot':
                y.append(self.rot_linear[cls_id](h[ins_i:ins_i+1]))
            else:
                raise NotImplementedError
        y = torch.cat(y, dim=0)
        return y

    def forward_by_mask(self, h, cls, out=None, phase=None):
        if out == 'y':
            if opt.reg is not None:
                y_all = [self.y_linear[i](h) for i in range(self.n_sem)]
            else:
                y_all = [self.y_linear[i](h) for i in range(self.n_sem)]
        elif out == 'rot':
            y_all = [self.rot_linear[i](h) for i in range(self.n_sem)]
        else:
            raise NotImplementedError
        if opt.sem_select == 'gt' and phase == 'train' and opt.refine == None and opt.bi_lstm == None:
            cls_id_all = cls
        else:
            _, cls_id_all = torch.max(cls, 1)
        mask_all = [cls_id_all == i for i in range(self.n_sem)]
        y = 0
        for i in range(self.n_sem):
            y += y_all[i] * mask_all[i].unsqueeze(dim=1).expand(y_all[i].size()).float()
        return y

    def get_local_feature(self, features, prim_box_2d):
        local_features = []
        prim_box_2d_unnorm = prim_box_2d * opt.input_res    # 16, 4, 33
        for i in range(prim_box_2d_unnorm.size(0)):
            features_i = [features[j][i:i + 1] for j in range(len(features))]
            dim = features_i[0].size(-1)
            if 'dense' in opt.box2d_en:
                for ii in range(len(features_i)):
                    assert dim == features_i[ii].size(-1)
                    x = torch.arange(dim).unsqueeze(dim=0).expand(dim, -1).unsqueeze(dim=0)
                    y = torch.arange(dim).unsqueeze(dim=1).expand(-1, dim).unsqueeze(dim=0)
                    dense_map = torch.cat((x,y), dim=0).type(torch.FloatTensor).unsqueeze(dim=0).cuda()
                    features_i[ii] = torch.cat((features_i[ii], dense_map), dim=1)
            prim_box_2d_i = prim_box_2d_unnorm[i].transpose(0, 1)   # 33, 4
            center_i = [(prim_box_2d_i[0, 0] + prim_box_2d_i[0, 2]) / 2 / opt.input_res * dim,
                        (prim_box_2d_i[0, 1] + prim_box_2d_i[0, 3]) / 2 / opt.input_res * dim]
            prim_boxlist_i = [BoxList(prim_box_2d_i, (opt.input_res, opt.input_res))]
            local_feature_i = self.pooler(features_i, prim_boxlist_i)   # 33, 256/258, 7, 7
            if opt.box2d_en == 'dense_norm':
                local_feature_i[:, -1, :, :] -= center_i[0]
                local_feature_i[:, -2, :, :] -= center_i[1]
            local_feature_i = self.conv_local(local_feature_i)  # 33, box2d_size, 1, 1
            local_feature_i = local_feature_i.view(local_feature_i.size(0), -1)
            local_feature_i = local_feature_i.transpose(0, 1)
            if opt.box2d_en == 'hard':
                local_feature_i = torch.cat((local_feature_i, prim_box_2d[i]), dim=0)
            local_feature_i = local_feature_i.unsqueeze(dim=0)
            local_features.append(local_feature_i)  # (32, 33)
        local_features = torch.cat(local_features, dim=0)   # (16, 32, 33)
        if 'local' in opt.feature_scale:
            local_features = local_features / local_features.norm(dim=1).unsqueeze(dim=1)
        elif 'ten' in opt.feature_scale:
            local_features /= 10
        return local_features

