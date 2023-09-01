import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import copy
import math
import scipy.io
from torch.autograd import Variable
import pdb

# from lib.models import define_resnet_encoder
from .bbox_model import define_resnet_encoder
from .hg_model import HGNet
from .graph_model import GATPrim, GraphNet, PrimMLPs
from lib.utils.prnn_utils import *
from lib.opts import *
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.structures.bounding_box import BoxList


class PrimGraphModel(nn.Module):
    def __init__(self):
        super(PrimGraphModel, self).__init__()
        global opt
        opt = get_opt()
        self.prnn_model_forward = PRNNModel(inverse=False)
        self.prnn_model_backward = PRNNModel(inverse=True)
        self.refine_module = nn.ModuleList([GraphNet(opt.node, opt.embed_type)] +
                                           [GraphNet(opt.refine_node, opt.refine_embed)
                                            for _ in range(opt.refine_mp - 1)])
        if opt.sem_final:
            self.prim_mlps = PrimMLPs()

    def forward(self, input_mat, rot_mat, cls_mat=None, depth=None, bbox=None, existence=None,
                prim_box_2d=None, length=None, item_id=None, epoch=-1, c_mask_mat=None, phase=None, length_i=None):
        # proposal_f, _ = self.prnn_model_forward(input_mat, rot_mat, cls_mat, depth, bbox, existence,
        #                                         prim_box_2d, length, item_id, epoch, c_mask_mat, phase)
        # proposal_b, _ = self.prnn_model_backward(input_mat, None, None, depth, None, None,
        #                                          None, length, item_id, epoch, c_mask_mat, phase)
        proposal_f, _ = self.prnn_model_forward(input_mat, rot_mat, cls_mat, depth, bbox, existence,
                                                prim_box_2d, length, item_id, epoch, c_mask_mat, phase, length_i)
        proposal_b, _ = self.prnn_model_backward(input_mat, rot_mat, cls_mat, depth, None, None,
                                                 prim_box_2d, length, item_id, epoch, c_mask_mat, phase, length_i)
        if opt.demo is None:
            outputs, targets = self.train_val(proposal_f, proposal_b, input_mat.size(0))
        else:
            outputs, targets = self.test(proposal_f, proposal_b)
        return outputs, targets

    def get_proposal_batch(self, b, proposal, inv=False):
        proposal['length'] = proposal['length'][b:b + 1]
        # proposal['input_mat'] = proposal['input_mat'][b:b + 1]
        proposal['features'] = [x[b:b + 1] for x in proposal['features']]
        proposal['c_mask_mat'] = proposal['c_mask_mat'][b:b + 1]
        if opt.loss_box2d is None:
            proposal['outputs'] = [tuple((x[0], x[1], x[2], torch.Tensor(x[0].size(0), 1).cuda()))
                                   for x in proposal['outputs']]
        proposal['outputs'] = [tuple(y[b:b + 1] for y in x) for x in proposal['outputs']]
        for key, value in proposal['prev_h'].items():
            proposal['prev_h'][key] = collections.OrderedDict({x: y[b:b + 1] for x, y in value.items()})
        if not inv:
            proposal['targets'] = [tuple(y[b:b + 1] for y in x) for x in proposal['targets']]
            gt_length = torch.sum(proposal['c_mask_mat']).int().item()
            proposal['targets'] = proposal['targets'][:gt_length]

        proposal['features'] = proposal['features'][:proposal['length'][0, 0].item()]
        proposal['outputs'] = proposal['outputs'][:proposal['length'][0, 0].item()]
        for i in range(len(proposal['prev_h'].keys())):
            if i > proposal['length'][0, 0].item():
                proposal['prev_h'].pop(i)
        return proposal

    def train_val(self, proposal_f, proposal_b, batch_size):
    # def train_val(self, input_mat, rot_mat, cls_mat, depth, bbox, existence,
    #               prim_box_2d, length, item_id, epoch, c_mask_mat, phase):
        # proposal_f, _ = self.prnn_model_forward(input_mat, rot_mat, cls_mat, depth, bbox, existence,
        #                                         prim_box_2d, length, item_id, epoch, c_mask_mat, phase)
        # proposal_b, _ = self.prnn_model_backward(input_mat, None, None, depth, None, None,
        #                                          None, length, item_id, epoch, c_mask_mat, phase)
        outputs_out, targets_out = [], []
        for b in range(batch_size):
            proposal_f_b = self.get_proposal_batch(b, copy.deepcopy(proposal_f))
            proposal_b_b = self.get_proposal_batch(b, copy.deepcopy(proposal_b), inv=True)
            proposal = combine_bi_direction(proposal_f_b, proposal_b_b)
            max_length, length, input_mat, \
            features, c_mask_mat, outputs, \
            prev_h, prev_c, targets = proposal['max_length'], proposal['length'], proposal['input_mat'], \
                                      proposal['features'], proposal['c_mask_mat'], proposal['outputs'], \
                                      proposal['prev_h'], proposal['prev_c'], proposal['targets']
            # print(b, length, len(features), len(outputs), len(prev_h.keys()), len(targets))
            outputs = self.refine_train(length, features, c_mask_mat, outputs, prev_h, targets)
            # print(b, length, len(features), len(outputs), len(prev_h.keys()), len(targets))
            outputs_out.append(outputs)
            targets_out.append(targets)
        return outputs_out, targets_out

    def refine_train(self, length, features, c_mask_mat, outputs, prev_h, targets):
        # import pdb;pdb.set_trace()
        assert opt.loss_e is None  # otherwise y and x are not the same
        outputs_refine = []
        if opt.len_source != 'None':
            lengths = length[:, 0].int().tolist()
        else:
            raise NotImplementedError
            # lengths = get_sequence_len_all(input_mat)
        nodes = prev_h
        for i in range(opt.refine_mp):
            # print(length, len(features), len(outputs), len(nodes), len(targets))
            outputs, nodes = self.refine_module[i](lengths, outputs, prev_h=nodes,
                                                   features=features, targets=targets, phase='train')
            # if opt.out_refine != 'None':
            #     outputs = self.refine_output(lengths, outputs, prev_h=prev_h, features=features)
            outputs_refine.append(outputs)
        if opt.sem_final:
            # print(length, len(features), len(outputs), len(nodes), len(targets))
            cls_gt_all, res_gt_id_all = get_matched_cls_gt(outputs, targets, c_mask_mat, length)
            outputs = self.prim_mlps(lengths, outputs, nodes, cls_gt_all, phase='train')
            outputs_refine.append((outputs, res_gt_id_all))
        return outputs_refine

    def test(self, proposal_f, proposal_b):
        proposal = combine_bi_direction(copy.deepcopy(proposal_f), copy.deepcopy(proposal_b))
        length, features, outputs, prev_h = proposal['length'], proposal['features'], \
                                            proposal['outputs'], proposal['prev_h']
        # print(length[0, 0].item(), len(features), len(outputs), len(prev_h.keys()))
        outputs = self.refine_test(features, outputs, prev_h, length[0, 0].item())
        # print(length[0, 0].item(), len(features), len(outputs), len(prev_h.keys()))
        outputs = outputs[-1]
        if opt.loss_box2d is None:
            outputs = [(out[0].cpu().detach(), out[1].cpu().detach(), out[2].cpu().detach(), torch.Tensor(out[0].size(0), 1))
                       for out in outputs]
        else:
            outputs = [(out[0].cpu().detach(), out[1].cpu().detach(), out[2].cpu().detach(), out[3].cpu().detach())
                       for out in outputs]
        return outputs, None

    def refine_test(self, features, outputs, prev_h, length_i=None):
        assert opt.loss_e is None  # otherwise y and x are not the same
        outputs_refine = []
        lengths = [int(length_i)]
        if opt.reg_init != 'None':
            nodes = prev_h
            for i in range(opt.refine_mp):
                # print(length_i, len(features), len(outputs), len(nodes))
                outputs, nodes = self.refine_module[i](lengths, outputs, prev_h=nodes, features=features, phase='val')
                if opt.out_refine != 'None':
                    outputs = self.refine_output(lengths, outputs, prev_h=prev_h, features=features)
                outputs_refine.append(outputs)
            if opt.sem_final:
                # print(length_i, len(features), len(outputs), len(nodes))
                outputs = self.prim_mlps(lengths, outputs, nodes, None, phase='val')
                outputs_refine.append(outputs)
        else:
            init = outputs[0]
            outputs = outputs[1:]
            nodes = prev_h
            for i in range(opt.refine_mp):
                outputs, nodes = self.refine_module[i](lengths, outputs, prev_h=nodes, features=features,
                                                       phase='val')
                if opt.out_refine != 'None':
                    outputs = self.refine_output(lengths, outputs, prev_h=prev_h, features=features)
                outputs_refine.append(outputs)
            if opt.sem_final:
                outputs = self.prim_mlps(lengths, outputs, nodes, None, phase='val')
                outputs_refine.append(outputs)
            for i in range(len(outputs_refine)):
                outputs_refine[i] = [init] + outputs_refine[i]

            # outputs_refine = [init] + self.refine_module(lengths, outputs[1:], prev_h=prev_h, features=features, phase='val')
            # outputs_refine = [outputs_refine]
        return outputs_refine


class PRNNModel(nn.Module):
    def __init__(self, inverse):
        super(PRNNModel, self).__init__()
        global opt
        opt = get_opt()
        self.root_dir = opt.data_dir
        self.inverse = inverse
        if opt.encoder == 'resnet':
            self.rgb_encoder = define_resnet_encoder(num_reg=opt.con_size)
        if opt.encoder == 'hg':
            self.rgb_encoder = HGNet(num_reg=opt.con_size)
        if opt.refine == 'gat':
            graph_in_node_size = opt.con_size + opt.box2d_size + opt.con_size
            self.refine_module = GATPrim(nfeat=graph_in_node_size, nhid=opt.n_hid_g,
                                     nclass=opt.n_out_g, dropout=opt.drop_g,
                                     nheads=opt.n_head_g, alpha=opt.alpha_g)
        elif opt.refine == 'prnn':
            self.refine_module = PRNNCore()
        elif opt.refine == 'graph':
            self.refine_module = nn.ModuleList([GraphNet(opt.node, opt.embed_type)] +
                                               [GraphNet(opt.refine_node, opt.refine_embed)
                                                for _ in range(opt.refine_mp - 1)])
            if opt.sem_final:
                self.prim_mlps = PrimMLPs()
        if opt.out_refine != 'None':
            self.refine_output = GraphNet(opt.out_refine)
        if opt.encoder in ['depth', 'depth_new']:
            self.depth_encoder = VOXAEnet()
        self.prnn_core = PRNNCore()
        if opt.fpn:
            # scales_all = [1/4, 1/8, 1/16, 1/32]
            scales_all = [1/4]
            self.pooler = Pooler(output_size=(7, 7), scales=scales_all, sampling_ratio=2)
            in_size = 256
            if 'dense' in opt.box2d_en:
                in_size += 2
            self.conv_local = nn.Conv2d(in_size, abs(opt.box2d_size), kernel_size=7, stride=1, padding=0)

    def compute_max_len(self, input_mat):
        input_max_len = input_mat.size()[2]
        for i in range(input_max_len, 0, -1):
            if torch.sum(torch.abs(input_mat[:, :, i - 1])) != 0:
                return i
        return 0

    def cls_to_prob(self, cls_mat):
        c_in = self.label_to_prob(cls_mat, dim=opt.n_sem)
        return c_in

    def rot_to_prob(self, rot_mat):
        theta = self.label_to_prob(rot_mat[:, 0:1], 27)
        axis = self.label_to_prob(rot_mat[:, 1:2], 4)
        rot_in = torch.cat((theta, axis), dim=1)
        return rot_in

    def label_to_prob(self, label, dim):
        eps = 1e-5
        batch_size = label.size(0)
        out = torch.zeros(batch_size, dim) + eps
        for i in range(batch_size):
            out[i, int(label[i, 0].item())] += (1. - eps * dim)
        if opt.use_gpu:
            out = out.cuda()
        return out

    def get_out_for_bbox_loss(self, pred_out, x_in, r_in, cls_mat, y, rot, cls, i):
        if i == 0:
            pred_out.append((x_in, r_in, self.cls_to_prob(cls_mat[:, :, 0])))
        if opt.bbox_loss == 'cls_ora':
            pred_out.append((self.get_x_fast(y), rot, self.cls_to_prob(cls_mat[:, :, i + 1])))
        elif opt.bbox_loss == 'cls_pre':
            pred_out.append((self.get_x_fast(y), rot, cls))
        else:
            pred_out.append((self.get_x_fast(y), rot, None))
        return pred_out

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
            local_features.append(local_feature_i)  # (128, 33)
        local_features = torch.cat(local_features, dim=0)
        if 'local' in opt.feature_scale:
            local_features = local_features / local_features.norm(dim=1).unsqueeze(dim=1)
        elif 'ten' in opt.feature_scale:
            local_features /= 10
        return local_features

    def compute_adj(self, batch_size, num_node, c_mask_mat=None):
        if opt.adj == 'lstm':
            adj = torch.zeros(batch_size, num_node, num_node).cuda()
            for graph_i in range(batch_size):
                for node_i in range(num_node):
                    adj[graph_i, node_i, node_i] = 1
                    if node_i - 1 >= 0:
                        adj[graph_i, node_i, node_i - 1] = 1
                # print(graph_i, adj[graph_i])
        elif 'xxyyzz' in opt.adj:
            adj = torch.zeros(batch_size, num_node, num_node).cuda()
            for graph_i in range(batch_size):
                for node_i in range(num_node):
                    adj[graph_i, node_i, node_i] = 1
                    if 'before' in opt.adj:
                        if node_i - 3 >= 0:
                            adj[graph_i, node_i, node_i - 3] = 1
                        if 'dense' in opt.adj:
                            if node_i - 2 >= 0:
                                adj[graph_i, node_i, node_i - 2] = 1
                            if node_i - 1 >= 0:
                                adj[graph_i, node_i, node_i - 1] = 1
                    if 'b2s' in opt.adj:
                        if node_i - 6 >= 0:
                            adj[graph_i, node_i, node_i - 6] = 1
                    if 'after' in opt.adj:
                        if node_i + 3 < num_node:
                            adj[graph_i, node_i, node_i + 3] = 1
                        if 'dense' in opt.adj:
                            if node_i + 2 < num_node:
                                adj[graph_i, node_i, node_i + 2] = 1
                            if node_i + 1 < num_node:
                                adj[graph_i, node_i, node_i + 1] = 1
                    if 'a2s' in opt.adj:
                        if node_i + 6 < num_node:
                            adj[graph_i, node_i, node_i + 6] = 1
                # print(graph_i, adj[graph_i])
        else:
            adj = torch.ones(batch_size, num_node, num_node).cuda()
            for graph_i in range(batch_size):
                sum = int(torch.sum(c_mask_mat[graph_i]).item())
                adj[graph_i, sum:, :] = 0
                adj[graph_i, :, sum:] = 0
                # print(graph_i, adj[graph_i])
        return adj

    def get_target(self, i, input_mat, rot_mat, cls_mat, prim_box_2d, targets):
        if self.inverse and opt.bi_lstm == 'online':
            targets.append((None, None, None, None))
            return targets
        if 'share' in opt.reg_init:
            i -= 1
        if i < input_mat.size(2):   #####max_length
            x_target = copy.deepcopy(input_mat[:, :, i])
            r_target = copy.deepcopy(rot_mat[:, :, i])
            c_target = copy.deepcopy(cls_mat[:, 0, i])
            b_target = copy.deepcopy(prim_box_2d[:, :, i])
            targets.append((x_target, r_target, c_target, b_target))
        return targets

    def append_init(self, d, outputs, targets, input_mat, rot_mat, cls_mat, prim_box_2d):
        # img_f, y, rot, cls, box2d = None, None, None, None, None
        img_f = None
        if 'idv' in opt.reg_init:
            d, x_init = d
            y, rot, cls, box2d = x_init
            # x_target, r_target, c_target, b_target = self.get_target(0, input_mat, rot_mat, cls_mat, prim_box_2d)
            outputs.append((y, rot, cls, box2d))
            targets = self.get_target(0, input_mat, rot_mat, cls_mat, prim_box_2d, targets)
            # targets.append((x_target, r_target, c_target, b_target))
        if 'share' in opt.reg_init:
            d, img_f = d
        return d, img_f, outputs, targets

    def init_train(self, i, epoch, outputs, input_mat, rot_mat, cls_mat, prim_box_2d, phase):
        if 'share' in opt.reg_init:
            i -= 1
        if len(outputs) > 0:
            y, rot, cls, box2d = outputs[-1]
        else:
            y, rot, cls, box2d = None, None, None, None
        if 'idv' in opt.reg_init and i == 0:
            x_in, r_in, c_in, b_in = outputs[0]
        elif 'share' in opt.reg_init and i == -1:   #i == 0:
            x_in, r_in, c_in, b_in = None, None, None, None
            if opt.step_in == 'detach' and opt.loss_c is None:
                if phase == 'train' or phase == 'train_val':
                    batch_size = opt.train_batch
                elif phase == 'val':
                    batch_size = opt.valid_batch
                else:
                    raise NotImplementedError
                c_in = self.cls_to_prob(torch.zeros(batch_size, 1).cuda())   # should be of no use
        elif (epoch < opt.step_in_ep or opt.step_in is None or (i == 0 and 'share' not in opt.reg_init)) \
                and i < input_mat.size(2):  #####max_length
            x_in = copy.deepcopy(input_mat[:, :, i])
            if opt.out_r == 'class':
                r_in = self.rot_to_prob(rot_mat[:, :, i])
            else:
                r_in = copy.deepcopy(rot_mat[:, :, i])
            c_in = self.cls_to_prob(cls_mat[:, :, i])
            b_in = copy.deepcopy(prim_box_2d[:, :, i])
            # pdb.set_trace()
        else:
            x_in = torch.zeros(y.size(0), opt.xyz * 2).cuda()
            # x_in = torch.zeros(input_mat[:, :, 0].size()).cuda()
            if opt.step_in == 'detach':
                for b_i in range(y.size()[0]):
                    x_in[b_i] = self.get_x_fast(y[b_i:b_i + 1])
                x_in = x_in.detach()
                r_in = rot.detach()
                if opt.loss_c is not None:
                    c_in = cls.detach()
                else:
                    c_in = self.cls_to_prob(torch.zeros(y.size(0), 1).cuda())   # should be of no use
                if opt.loss_box2d is not None:
                    b_in = box2d.detach()
                else:
                    b_in = box2d
            elif opt.step_in == 'in':
                for b_i in range(y.size()[0]):
                    x_in[b_i] = self.get_x_fast(y[b_i:b_i + 1])
                r_in = rot
                c_in = cls
                b_in = box2d
            else:
                raise NotImplementedError
        return x_in, r_in, c_in, b_in

    def get_local_encoding(self, i, global_encoding, local_features, prim_box_2d, box2d, phase):
        if opt.box2d_source == 'oracle':
            local_encoding = local_features[:, :, i]    # 16, 32
        elif 'pred' in opt.box2d_source:
            if i == 0 and 'share' in opt.reg_init:
                box2d_pred = torch.ones(local_features[0].size(0), 4).cuda() * 0.9
                # box2d_pred = torch.ones(prim_box_2d[:, :, i].size()).cuda() * 0.9
                box2d_pred[:, :2] = 0.1
            elif i == 0 and opt.reg_init == 'None' and phase == 'train':
                box2d_pred = copy.deepcopy(prim_box_2d[:, :, i])
            else:
                if 'detach' in opt.box2d_source:
                    box2d_pred = box2d.detach()
                else:
                    box2d_pred = box2d
            local_encoding = self.get_local_feature(local_features, box2d_pred.unsqueeze(dim=2))
            local_encoding = local_encoding[:, :, 0]    # 16, 32
        else:
            raise NotImplementedError
        if opt.box2d_pos == '0':
            if opt.box2d_size > 0:
                d = torch.cat((global_encoding, local_encoding), dim=1)
            else:
                d = global_encoding + local_encoding
            local_encoding = None
        else:
            d = global_encoding
        return d, local_encoding

    def train_val(self, input_mat, rot_mat, cls_mat, prim_box_2d, length, item_id, d, prev_h, prev_c,
                  epoch, c_mask_mat, phase):
        # step_in, step_in_ep = opt.step_in, opt.step_in_ep
        # if opt.val_loss and phase == 'val':
        #     opt.step_in, opt.step_in_ep = 'detach', 0
        input_max_len = self.compute_max_len(input_mat)
        if opt.inverse and opt.bi_lstm == 'online':
            input_mat, rot_mat, cls_mat, prim_box_2d = None, None, None, None
        # print(input_max_len)
        max_length = min(input_max_len, opt.max_len)
        seq_length = max_length - 1 # initial prim given
        features, outputs, targets, pred_out = [], [], [], []
        img_f, y, rot, cls, box2d = None, None, None, None, None
        global_encoding, local_features, local_encoding = None, None, None
        if opt.reg_init != 'None':
            d, img_f, outputs, targets = self.append_init(d, outputs, targets, input_mat, rot_mat, cls_mat, prim_box_2d)
            if 'share' in opt.reg_init:
                seq_length = max_length
        if opt.len_adjust:
            # assert 'share' in opt.reg_init
            pre_length = torch.max(length).int().item()
            if opt.graph_xyz == 3 and opt.xyz == 1:
                pre_length *= 3
            if seq_length < pre_length: # pred length longer than gt length
                seq_length = pre_length
            # if 'share' in opt.reg_init:
        else:
            pre_length = seq_length
        if opt.fpn:
            global_encoding, local_features = d
            if opt.box2d_source == 'oracle':
                local_features = self.get_local_feature(local_features, prim_box_2d) # prim_box_2d.size() (16, 4, 33)
        for i in range(seq_length):
            targets = self.get_target(i + 1, input_mat, rot_mat, cls_mat, prim_box_2d, targets)
            # x_target, r_target, c_target, b_target = self.get_target(i + 1, input_mat, rot_mat, cls_mat, prim_box_2d)
            # targets.append((x_target, r_target, c_target, b_target))
        for i in range(seq_length):
            x_in, r_in, c_in, b_in = self.init_train(i, epoch, outputs, input_mat, rot_mat, cls_mat, prim_box_2d, phase)
            if opt.fpn:
                if i == 0 and 'idv' in opt.reg_init:
                    box2d = b_in
                if (i + opt.step_start + 2) % (3 // opt.xyz) == 0 or i == 0:
                    d, local_encoding = self.get_local_encoding(i, global_encoding, local_features,
                                                               prim_box_2d, box2d, 'train')
            # x_target, r_target, c_target, b_target = self.get_target(i + 1, input_mat, rot_mat, cls_mat, prim_box_2d)
            y, rot, cls, box2d, prev_h[i + 1], prev_c[i + 1] = self.prnn_core(
                x_in, r_in, cls=c_in, box2d=b_in, d=d, local=local_encoding, img_f=img_f,
                prev_h=prev_h[i], prev_c=prev_c[i], step=i, targets=targets, phase='train')
            if i < pre_length:
                features = self.append_feature(features, img_f, global_encoding, d)
                outputs.append((y, rot, cls, box2d))
            # targets.append((x_target, r_target, c_target, b_target))
        if opt.bi_lstm == 'online' or opt.proposal == 'save':
            for i in range(len(prev_h.keys())):
                if i > pre_length:
                    prev_h.pop(i)
                    prev_c.pop(i)
            assert len(prev_h.keys()) == len(outputs) + 1
        if opt.lstm_prop == 'gmm':
            c_mask_mat, outputs, targets, prev_h = self.pack_up_for_graph(c_mask_mat, outputs, targets, prev_h)
        if opt.refine is not None:
            if opt.proposal == 'save':
                self.save_proposal(item_id, max_length, length, input_mat, features, c_mask_mat, outputs,
                                   prev_h, prev_c, targets, phase)
            outputs = self.refine_train(max_length, length, input_mat, features, c_mask_mat,
                                        outputs, prev_h, prev_c, targets)
        # outputs.append(pred_out)
        # if opt.val_loss and phase == 'val':
        #     opt.step_in, opt.step_in_ep = step_in, step_in_ep
        if opt.bi_lstm == 'online':
            # print(length, len(features), len(outputs), len(prev_h.keys()), len(targets))
            proposal = {}
            proposal['max_length'], proposal['length'], proposal['input_mat'], \
            proposal['features'], proposal['c_mask_mat'], proposal['outputs'], \
            proposal['prev_h'], proposal['prev_c'], proposal['targets'] = max_length, length, input_mat, \
                                                                          features, c_mask_mat, outputs, \
                                                                          prev_h, prev_c, targets
            return proposal, None
        return outputs, targets

    def pack_up_for_graph(self, c_mask_mat, outputs, targets, prev_h):
        pdb.set_trace()
        batch_size = c_mask_mat.size(0)
        length = c_mask_mat.size(2) // 3
        c_mask_mat_new = torch.zeros(c_mask_mat.size(0), c_mask_mat.size(1), length).cuda()
        for b in range(batch_size):
            idx = int(torch.sum(c_mask_mat[b]).item()) // 3
            c_mask_mat_new[b, :, :idx] = 1
        pdb.set_trace()
        outputs_new = []
        for item in outputs:
            y, rot, cls, box2d = item
            tmp_x = torch.zeros(y.size(0), opt.xyz * 2).cuda()
            for b_i in range(y.size()[0]):
                tmp_x[b_i] = self.get_x_fast(y[b_i:b_i + 1])
            outputs_new.append((tmp_x, rot, cls, box2d))
        outputs = outputs_new
        pdb.set_trace()
        targets_new = []
        len_tar = len(targets) // 3
        for i in range(0, len(targets), 3):
            x = [item[0] for item in targets[i:i + 3]]
            r = [item[1] for item in targets[i:i + 3]]
            x_i = torch.cat((x[0][:, 0:1], x[1][:, 0:1], x[2][:, 0:1], x[0][:, 1:], x[1][:, 1:], x[2][:, 1:]), dim=1)
            r_i = torch.cat(r, dim=1)
            c_i = targets[i + 2 * (opt.reg_init == 'None')][2]
            b_i = targets[i + 2 * (opt.reg_init == 'None')][3]
            targets_new.append((x_i, r_i, c_i, b_i))

        prev_h_new = []
        return c_mask_mat_new, outputs_new, targets_new, prev_h_new

    def xyz_one_step(self, input_mat, rot_mat, cls_mat, c_mask_mat, max_length, ins_length):
        length = max_length // 3
        input_mat_new = torch.zeros(2 * opt.xyz + 1, length)
        rot_mat_new = torch.zeros(2 * opt.xyz, length)
        cls_mat_new = torch.zeros(1, length)
        c_mask_mat_new = torch.zeros(1, length)
        c_mask_mat_new[:, :ins_length // 3] = 1.
        for i in range(0, max_length, 3):
            input_mat_new[0:3, i // 3] = input_mat[0, i:i + 3]
            input_mat_new[3:6, i // 3] = input_mat[1, i:i + 3]
            input_mat_new[6, i // 3] = input_mat[2, i + 2]  # 0 0 0 0 0 1
            rot_mat_new[0:3, i // 3] = rot_mat[0, i:i + 3]
            rot_mat_new[3:6, i // 3] = rot_mat[1, i:i + 3]
            cls_mat_new[0, i // 3] = cls_mat[0, i]  # 0 0 0 1 1 1
        cls_mat_new = cls_mat_new.long()
        if ins_length == 0:
            input_mat_new[6, 0] = 1
        return input_mat_new, rot_mat_new, cls_mat_new, c_mask_mat_new

    def save_proposal(self, item_id, max_length, length, input_mat, features, c_mask_mat, outputs,
                      prev_h, prev_c, targets, phase):
        proposal = collections.OrderedDict()
        proposal['max_length'] = max_length  # scalar
        proposal['length'] = length  # (1,1) int
        proposal['input_mat'] = input_mat  # (6, *)
        proposal['features'] = features  # list - tensor
        proposal['c_mask_mat'] = c_mask_mat  # (1,1,*)
        proposal['outputs'] = outputs  # list - tuple - tensor
        proposal['prev_h'] = prev_h  # dict - OrderedDict - tensor
        proposal['prev_c'] = prev_c  # dict - OrderedDict - tensor
        proposal['targets'] = targets  # list - tuple - tensor
        proposal_name = phase
        if self.inverse:
            proposal_name = '{}_inverse'.format(phase)
        save_path = os.path.join(self.root_dir, 'proposals')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = os.path.join(save_path, proposal_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_file_pth = os.path.join(save_path, '{}.pth'.format(item_id[0, 0].item()))
        torch.save(proposal, save_file_pth)
        self.proposal_gpu_to_cpu(save_file_pth)

    def proposal_gpu_to_cpu(self, save_file_pth):
        proposal = torch.load(save_file_pth)
        proposal['length'] = proposal['length'].cpu()
        proposal['input_mat'] = proposal['input_mat'].cpu()
        proposal['features'] = [x.cpu() for x in proposal['features']]
        proposal['c_mask_mat'] = proposal['c_mask_mat'].cpu()
        proposal['outputs'] = [tuple(y.cpu() for y in x) for x in proposal['outputs']]
        proposal['prev_h'] = {i: collections.OrderedDict({j: y.cpu() for (j, y) in x.items()})
                              for (i, x) in proposal['prev_h'].items()}
        proposal['prev_c'] = {i: collections.OrderedDict({j: y.cpu() for (j, y) in x.items()})
                              for (i, x) in proposal['prev_c'].items()}
        proposal['targets'] = [tuple(y.cpu() for y in x) for x in proposal['targets']]
        torch.save(proposal, save_file_pth)

    def load_proposals_all(self, data_sizes):
        self.proposal_all = {'train': [], 'val': []}
        for phase, size in data_sizes.items():
            for item_id in range(size):
                proposal = self.load_proposal_to_gpu(item_id, phase, inverse=self.inverse)
                if opt.bi_lstm == 'offline':
                    proposal_inv = self.load_proposal_to_gpu(item_id, phase, inverse=True)
                    proposal = combine_bi_direction(proposal, proposal_inv)
                self.proposal_all[phase].append(proposal)

    def load_proposal_to_gpu(self, item_id, phase, inverse=False):
        proposal_name = phase
        if inverse:
            proposal_name = '{}_inverse'.format(phase)
        save_path = os.path.join(self.root_dir, 'proposals', proposal_name)
        save_file_pth = os.path.join(save_path, '{}.pth'.format(item_id))
        proposal = torch.load(save_file_pth)
        proposal['length'] = proposal['length'].cuda()
        proposal['input_mat'] = proposal['input_mat'].cuda()
        proposal['features'] = [x.cuda() for x in proposal['features']]
        proposal['c_mask_mat'] = proposal['c_mask_mat'].cuda()
        proposal['outputs'] = [tuple(y.cuda() for y in x) for x in proposal['outputs']]
        proposal['prev_h'] = {i: collections.OrderedDict({j: y.cuda() for (j, y) in x.items()})
                              for (i, x) in proposal['prev_h'].items()}
        proposal['prev_c'] = {i: collections.OrderedDict({j: y.cuda() for (j, y) in x.items()})
                              for (i, x) in proposal['prev_c'].items()}
        proposal['targets'] = [tuple(y.cuda() for y in x) for x in proposal['targets']]
        return proposal

    def train_val_proposal(self, item_id, phase):
        proposal = self.proposal_all[phase][item_id]
        max_length, length, input_mat, \
        features, c_mask_mat, outputs, \
        prev_h, prev_c, targets = proposal['max_length'], proposal['length'], proposal['input_mat'], \
                                  proposal['features'], proposal['c_mask_mat'], proposal['outputs'], \
                                  proposal['prev_h'], proposal['prev_c'], proposal['targets']
        #self.load_proposal_to_gpu(item_id, phase)
        outputs = self.refine_train(max_length, length, input_mat, features, c_mask_mat,
                                    outputs, prev_h, prev_c, targets)
        return outputs, targets

    def refine_train(self, max_length, length, input_mat, features, c_mask_mat, outputs, prev_h, prev_c, targets):
        # import pdb;pdb.set_trace()
        assert opt.loss_e is None  # otherwise y and x are not the same
        outputs_refine = []
        if opt.refine == 'prnn':
            # y, rot, cls, box2d = outputs[0]
            global_encoding, local_features, local_encoding = None, None, None
            refine_steps = max_length - 1
            if opt.reg_init != 'None':
                refine_steps = max_length
            if 'share' in opt.reg_init:
                raise NotImplementedError
            for i in range(refine_steps):
                x_in, r_in, c_in, b_in = outputs[i]
                # if opt.residual:
                #     x_in_ori, r_in_ori, c_in_ori, b_in_ori = x_in.detach(), r_in.detach(), c_in.detach(), b_in
                y, rot, cls, box2d, prev_h[i + max_length], prev_c[i + max_length] = self.refine_module(
                    x_in, r_in, cls=c_in, box2d=b_in, d=features[i], local=local_encoding,
                    prev_h=prev_h[i + max_length - 1], prev_c=prev_c[i + max_length - 1], step=i)
                if opt.residual:
                    outputs_refine.append((y + x_in, rot + r_in, cls + c_in, box2d + b_in))
                else:
                    outputs_refine.append((y, rot, cls, box2d))
                # targets.append((x_target, r_target, c_target, b_target))
        if opt.refine == 'gat':
            batch_size = opt.train_batch #input_mat.size(0)
            num_node = len(features)
            adj = self.compute_adj(batch_size, num_node, c_mask_mat)
            outputs_refine = self.refine_module(features, outputs, adj, phase='train')
        if opt.refine == 'graph':
            if opt.len_source != 'None':
                lengths = length[:, 0].int().tolist()
            else:
                raise NotImplementedError
                # lengths = get_sequence_len_all(input_mat)
            nodes = prev_h
            for i in range(opt.refine_mp):
                outputs, nodes = self.refine_module[i](lengths, outputs, prev_h=nodes,
                                                       features=features, targets=targets, phase='train')
                if opt.out_refine != 'None':
                    outputs = self.refine_output(lengths, outputs, prev_h=prev_h, features=features)
                outputs_refine.append(outputs)
            if opt.sem_final:
                cls_gt_all, res_gt_id_all = get_matched_cls_gt(outputs, targets, c_mask_mat, length)
                outputs = self.prim_mlps(lengths, outputs, nodes, cls_gt_all, phase='train')
                outputs_refine.append((outputs, res_gt_id_all))
        return outputs_refine

    def test_batch(self, i):
        path = '/Users/heqian/Research/projects/3dprnn/3d1/network'
        output_y = scipy.io.loadmat(os.path.join(path, 'save_output_y.mat'))
        rot_res = scipy.io.loadmat(os.path.join(path, 'save_rot_res.mat'))
        output_y = output_y['x']
        rot_res = rot_res['x']
        outputs = []
        # for i in range(opt.pred_len):
        #     tmp = (torch.from_numpy(output_y[i:i+1, :]), torch.from_numpy(rot_res[i:i+1, :]))
        #     outputs.append(tmp)
        tmp = (Variable(torch.from_numpy(output_y[i:i + 1, :]).float()).cuda(),
               Variable(torch.from_numpy(rot_res[i:i + 1, :]).float()).cuda())
        return tmp

    def test_gt(self, input_mat, rot_mat, cls_mat, prim_box_2d, length, d, prev_h, prev_c):
        len_gt = input_mat.size()[2]
        x_in = copy.deepcopy(input_mat[:, :, 0])
        r_in = copy.deepcopy(rot_mat[:, :, 0])
        c_in = self.cls_to_prob(cls_mat[:, :, 0])
        b_in = copy.deepcopy(prim_box_2d[:, :, 0])
        x_out = {0: x_in}
        r_out = {0: r_in}
        c_out = {0: c_in}
        b_out = {0: b_in}
        outputs = []
        for i in range(opt.pred_len):
            if i < len_gt:
                x_in = copy.deepcopy(input_mat[:, :, i])
                r_in = copy.deepcopy(rot_mat[:, :, i])
                c_in = self.cls_to_prob(cls_mat[:, :, i])
                b_in = copy.deepcopy(prim_box_2d[:, :, i])
            else:
                x_in = x_out[i]
                r_in = r_out[i]
                c_in = c_out[i]
                b_in = b_out[i]
            y, r_out[i + 1], c_out[i + 1], b_out[i + 1], prev_h[i + 1], prev_c[i + 1] = self.prnn_core(
                x_in, r_in, cls=c_in, box2d=b_in, d=d, prev_h=prev_h[i], prev_c=prev_c[i], step=i)
            # y, r_in[i + 1] = self.test_batch(i)
            x_out[i + 1] = self.get_x_fast(y)
            if opt.init_in is None:
                outputs.append((x_out[i].cpu().detach(), r_out[i].cpu().detach(),
                                c_out[i].cpu().detach(), b_out[i].cpu().detach(), y.cpu().detach()))
            else:
                outputs.append((x_out[i + 1].cpu().detach(), r_out[i + 1].cpu().detach(),
                                c_out[i + 1].cpu().detach(), b_out[i + 1].cpu().detach(), y.cpu().detach()))  ##### y is wrong step
            if d is None:
                print('d', opt.encoder, d)
        return outputs, None

    def test_loss(self, input_mat, rot_mat, cls_mat, prim_box_2d, length, d, prev_h, prev_c):
        if 'test_loss' in opt.check:
            import pdb
            pdb.set_trace()
            print('test_loss', input_mat, rot_mat, cls_mat, prim_box_2d)
        len_gt = input_mat.size()[2]
        input_max_len = self.compute_max_len(input_mat)
        max_length = min(input_max_len, opt.max_len)
        x_in = copy.deepcopy(input_mat[:, :, 0])
        if opt.out_r == 'class':
            r_in = self.rot_to_prob(rot_mat[:, :, 0])
        else:
            r_in = copy.deepcopy(rot_mat[:, :, 0])
        c_in = self.cls_to_prob(cls_mat[:, :, 0])
        b_in = copy.deepcopy(prim_box_2d[:, :, 0])
        x_out = {0: x_in}
        r_out = {0: r_in}
        c_out = {0: c_in}
        b_out = {0: b_in}
        outputs = []
        targets = []
        global_encoding, local_features, local_encoding = None, None, None
        if opt.fpn:
            global_encoding, local_features = d
            if opt.box2d_source == 'oracle':
                local_features = self.get_local_feature(local_features, prim_box_2d)
            b_in = {0: copy.deepcopy(prim_box_2d[:, :, 0])}
        for i in range(max_length):
            if opt.test_loss == 'test_gt_loss':
                x_in = copy.deepcopy(input_mat[:, :, i])
                if opt.out_r == 'class':
                    r_in = self.rot_to_prob(rot_mat[:, :, i])
                else:
                    r_in = copy.deepcopy(rot_mat[:, :, i])
                c_in = self.cls_to_prob(cls_mat[:, :, i])
                b_in = copy.deepcopy(prim_box_2d[:, :, i])
            else:
                x_in = x_out[i]
                r_in = r_out[i]
                c_in = c_out[i]
                b_in = b_out[i]
            if opt.fpn:
                if opt.box2d_source == 'oracle':
                    local_encoding = local_features[:, :, i] # 16, 32
                elif 'pred' in opt.box2d_source:
                    if i == 0:
                        box2d = copy.deepcopy(prim_box_2d[:, :, i])
                    else:
                        box2d = b_out[i]
                    local_encoding = self.get_local_feature(local_features, box2d.unsqueeze(dim=2))
                    local_encoding = local_encoding[:, :, 0] # 16, 32
                else:
                    raise NotImplementedError
                if opt.box2d_pos == '0':
                    if opt.box2d_size > 0:
                        d = torch.cat((global_encoding, local_encoding), dim=1)
                    else:
                        d = global_encoding + local_encoding
                    local_encoding = None
                else:
                    d = global_encoding
            if i + 1 < input_mat.size(2):
                x_target = copy.deepcopy(input_mat[:, :, i + 1])
                r_target = copy.deepcopy(rot_mat[:, :, i + 1])
                c_target = copy.deepcopy(cls_mat[:, 0, i + 1])
                b_target = copy.deepcopy(prim_box_2d[:, :, i + 1])
            else:
                x_target, r_target, c_target, b_target = None, None, None, None
            if 'test_loss' in opt.check and i < 5:
                print('in', x_in, r_in, c_in, b_in)
            y, r_out[i + 1], c_out[i + 1], b_out[i + 1], prev_h[i + 1], prev_c[i + 1] = self.prnn_core(
                x_in, r_in, cls=c_in, box2d=b_in, d=d, local=local_encoding,
                prev_h=prev_h[i], prev_c=prev_c[i], step=i)
            # y, r_in[i + 1] = self.test_batch(i)
            x_out[i + 1] = self.get_x_fast(y)
            if opt.init_in is None:
                if opt.test_loss is not None:
                    outputs.append((x_out[i].detach(), r_out[i].detach(),
                                    c_out[i].detach(), b_out[i].detach(), y.detach()))
                    targets.append((x_target, r_target, c_target, b_target))
                else:
                    outputs.append((x_out[i].cpu().detach(), r_out[i].cpu().detach(),
                                    c_out[i].cpu().detach(), b_out[i].cpu().detach(), y.cpu().detach()))
            else:
                outputs.append((x_out[i + 1].cpu().detach(), r_out[i + 1].cpu().detach(),
                                c_out[i + 1].cpu().detach(), b_out[i + 1].cpu().detach(),
                                y.cpu().detach()))  ##### y is wrong step
            if 'test_loss' in opt.check:
                print('outputs', outputs[-1])
                print('targets', targets[-1])
            if d is None:
                print('d', opt.encoder, d)
        return outputs, targets

    def init_test(self, d, input_mat, rot_mat, cls_mat, prim_box_2d):
        if 'share' in opt.reg_init:
            d, img_f = d
            x_in = {0: None}
            if opt.out_r == 'class':
                r_in = {0: None}
            else:
                r_in = {0: None}
            c_in = {0: torch.ones(1, 1).cuda()}
            b_in = {0: None}
            return d, img_f, x_in, r_in, c_in, b_in
        img_f = None
        x_in = {0: copy.deepcopy(input_mat)}
        if opt.out_r == 'class':
            r_in = {0: self.rot_to_prob(rot_mat)}
        else:
            r_in = {0: copy.deepcopy(rot_mat)}
        c_in = {0: self.cls_to_prob(cls_mat)}
        b_in = {0: copy.deepcopy(prim_box_2d)}
        if opt.fpn:
            b_in = {0: copy.deepcopy(prim_box_2d[:, :, 0])}
            # replaced by nn prediction in dataset
        if 'idv' in opt.reg_init:
            d, x_init = d
            x_in = {0: x_init[0]}
            r_in = {0: x_init[1]}
            if opt.loss_c is not None:
                c_in = {0: x_init[2]}
            if opt.loss_box2d is not None:
                b_in = {0: x_init[3]}
        return d, img_f, x_in, r_in, c_in, b_in

    def append_feature(self, features, img_f, global_encoding, d):
        if 'img_f' in opt.node:
            features.append(img_f)
        elif 'img_global' in opt.node:
            features.append(global_encoding)
        else:
            features.append(d)
        return features

    def test(self, input_mat, rot_mat, cls_mat, prim_box_2d, length, d, prev_h, prev_c, length_i=None):
        d, img_f, x_in, r_in, c_in, b_in = self.init_test(d, input_mat, rot_mat, cls_mat, prim_box_2d)
        features, outputs = [], []
        global_encoding, local_features, local_encoding = None, None, None
        pred_len = opt.pred_len // opt.xyz + int(opt.xyz == 3)  # 14
        if opt.fpn:
            global_encoding, local_features = d
            if opt.box2d_source == 'oracle':
                local_features = self.get_local_feature(local_features, prim_box_2d)
                pred_len = local_features.size(2)
        for i in range(pred_len):
            if opt.fpn:
                box2d = b_in[i]
                d, local_encoding = self.get_local_encoding(i, global_encoding, local_features,
                                                            prim_box_2d, box2d, 'test')
            y, r_in[i + 1], c_in[i + 1], b_in[i + 1], prev_h[i + 1], prev_c[i + 1] = self.prnn_core(
                x_in[i], r_in[i], cls=c_in[i], box2d=b_in[i], d=d, local=local_encoding, img_f=img_f,
                prev_h=prev_h[i], prev_c=prev_c[i], step=i)
            # y, r_in[i + 1] = self.test_batch(i)
            x_in[i + 1] = self.get_x_fast(y)
            if opt.init_in is None and 'share' not in opt.reg_init:
                # features.append(d)
                features = self.append_feature(features, img_f, global_encoding, d)
                outputs.append((x_in[i], r_in[i], c_in[i], b_in[i]))
                # y tensor([[ 0.4657, -0.0570, -0.0100]])
                # x tensor([[-0.0570, -0.0100,  0.0000]])
            else:
                # features.append(d)
                features = self.append_feature(features, img_f, global_encoding, d)
                outputs.append((x_in[i + 1], r_in[i + 1], c_in[i + 1], b_in[i + 1]))
        if opt.refine is not None:
            outputs = self.refine_test(input_mat, features, outputs, prev_h, prev_c, length_i=length_i)
        if opt.refine == 'graph':
            outputs = outputs[-1]
        if opt.bi_lstm == 'online':
            pre_length = int(length_i)
            features = features[:int(length_i)]
            outputs = outputs[:int(length_i)]
            # print(length[0, 0].item(), int(length_i))
            for i in range(len(prev_h.keys())):
                if i > pre_length:
                    prev_h.pop(i)
                    prev_c.pop(i)
            assert len(prev_h.keys()) == len(outputs) + 1
            length = copy.deepcopy(length)  # just for the size
            length[0, 0] = pre_length
            # print(length[0, 0].item(), len(features), len(outputs), len(prev_h.keys()))
            proposal = {}
            proposal['length'], proposal['features'], proposal['outputs'], \
            proposal['prev_h'] = length, features, outputs, prev_h
            return proposal, None
        if opt.loss_c is None and opt.loss_box2d is None:
            outputs = [(out[0].cpu().detach(), out[1].cpu().detach(), None, None)
                       for out in outputs]
        elif opt.loss_c is None:
            outputs = [(out[0].cpu().detach(), out[1].cpu().detach(), None, out[3].cpu().detach())
                       for out in outputs]
        elif opt.loss_box2d is None:
            outputs = [(out[0].cpu().detach(), out[1].cpu().detach(), out[2].cpu().detach(), None)
                       for out in outputs]
        # if opt.loss_c is not None and opt.loss_box2d is not None:
        else:
            outputs = [(out[0].cpu().detach(), out[1].cpu().detach(), out[2].cpu().detach(), out[3].cpu().detach())
                       for out in outputs]
        return outputs, None

    def test_proposal(self, item_id, phase):
        proposal = self.proposal_all['val'][item_id]########
        max_length, length, input_mat, \
        features, c_mask_mat, outputs, \
        prev_h, prev_c, targets = proposal['max_length'], proposal['length'], proposal['input_mat'], \
                                  proposal['features'], proposal['c_mask_mat'], proposal['outputs'], \
                                  proposal['prev_h'], proposal['prev_c'], proposal['targets']
        #self.load_proposal_to_gpu(item_id, phase)
        print(length[0, 0].item(), len(features), len(outputs), len(prev_h.keys()))
        outputs = self.refine_test(input_mat, features, outputs, prev_h, prev_c, length[0, 0].item())
        if item_id < 10:
            print(item_id, length[0, 0].item())
        if opt.refine == 'graph':
            outputs = outputs[-1]
        outputs = [(out[0].cpu().detach(), out[1].cpu().detach(), out[2].cpu().detach(), out[3].cpu().detach())
                   for out in outputs]
        return outputs, targets

    def refine_test(self, input_mat, features, outputs, prev_h, prev_c, length_i=None):
        assert opt.loss_e is None  # otherwise y and x are not the same
        outputs_refine = []
        if opt.refine == 'prnn':
            if opt.reg_init:
                length = opt.pred_len
                start = 0
            elif opt.init_in is None:
                outputs_refine.append(outputs[0])
                length = opt.pred_len - 1
                start = 1
            else:
                length = opt.pred_len
                start = 0
            global_encoding, local_features, local_encoding = None, None, None
            for i in range(length):
                x_in, r_in, c_in, b_in = outputs[i + start]
                y, rot, cls, box2d, prev_h[i + opt.pred_len], prev_c[i + opt.pred_len] = self.refine_module(
                    x_in, r_in, cls=c_in, box2d=b_in, d=features[i], local=local_encoding,
                    prev_h=prev_h[i + opt.pred_len - 1], prev_c=prev_c[i + opt.pred_len - 1], step=i)
                if opt.residual:
                    outputs_refine.append((y + x_in, rot + r_in, cls + c_in, box2d + b_in))
                else:
                    outputs_refine.append((y, rot, cls, box2d))
        if opt.refine == 'gat':
            features.pop(-1)
            batch_size = opt.valid_batch #input_mat.size(0)
            num_node = len(features)
            adj = self.compute_adj(batch_size, num_node)
            init = outputs[0]
            outputs_refine = [init] + self.refine_module(features, outputs[1:], adj)
        if opt.refine == 'graph':
            lengths = [int(length_i)]
            if opt.reg_init != 'None':
                nodes = prev_h
                for i in range(opt.refine_mp):
                    outputs, nodes = self.refine_module[i](lengths, outputs, prev_h=nodes, features=features, phase='val')
                    if opt.out_refine != 'None':
                        outputs = self.refine_output(lengths, outputs, prev_h=prev_h, features=features)
                    outputs_refine.append(outputs)
                if opt.sem_final:
                    outputs = self.prim_mlps(lengths, outputs, nodes, None, phase='val')
                    outputs_refine.append(outputs)
            else:
                init = outputs[0]
                outputs = outputs[1:]
                nodes = prev_h
                for i in range(opt.refine_mp):
                    outputs, nodes = self.refine_module[i](lengths, outputs, prev_h=nodes, features=features,
                                                           phase='val')
                    if opt.out_refine != 'None':
                        outputs = self.refine_output(lengths, outputs, prev_h=prev_h, features=features)
                    outputs_refine.append(outputs)
                if opt.sem_final:
                    outputs = self.prim_mlps(lengths, outputs, nodes, None, phase='val')
                    outputs_refine.append(outputs)
                for i in range(len(outputs_refine)):
                    outputs_refine[i] = [init] + outputs_refine[i]

                # outputs_refine = [init] + self.refine_module(lengths, outputs[1:], prev_h=prev_h, features=features, phase='val')
                # outputs_refine = [outputs_refine]
        return outputs_refine

    def get_x_fast(self, y, i_component=None):
        # print(y)
        assert y.size()[0] == 1
        if opt.loss_e is None:
            x_in = torch.zeros(1, 2*opt.xyz)
            if opt.use_gpu:
                x_in = x_in.cuda()
        else:
            x_in = torch.zeros(1, 2*opt.xyz+opt.dim_e)
            if opt.use_gpu:
                x_in = x_in.cuda()
            e_t = y[0, 0:opt.dim_e]
            # print('e_t', e_t)
            if opt.no_rand:
                if e_t.item() > opt.stop_thresh:
                    x_in[0, -1] = e_t * 0 + 1
                else:
                    x_in[0, -1] = e_t * 0
            else:
                x_in[0, -1] = torch.bernoulli(e_t)
        if opt.loss_y == 'gmm' or opt.lstm_prop == 'gmm':
            pi_t = y[0, opt.dim_e:opt.dim_e + opt.n_component]
            mu_1_t = y[0, opt.dim_e + opt.n_component:opt.dim_e + opt.n_component * 2]
            mu_2_t = y[0, opt.dim_e + opt.n_component * 2:opt.dim_e + opt.n_component * 3]
            _, chosen_pi = torch.max(pi_t, 0)   # value, location
            x_in[0, 0] = mu_1_t[chosen_pi]
            x_in[0, 1] = mu_2_t[chosen_pi]
        elif opt.loss_y in ['l2', 'l2c', 'l1', 'l1c']:
            x_in[0, :opt.xyz] = y[0, opt.dim_e            :opt.dim_e + 1*opt.xyz]
            x_in[0, opt.xyz:opt.xyz*2] = y[0, opt.dim_e + 1*opt.xyz:opt.dim_e + 2*opt.xyz]
        else:
            raise NotImplementedError

        return x_in

    def get_x(self, y, i_component=None):
        # print(y)
        assert y.size()[0] == 1
        if opt.loss_e is None:
            x_in = torch.zeros(1, 2*opt.xyz)
            if opt.use_gpu:
                x_in = x_in.cuda()
        else:
            x_in = torch.zeros(1, 2*opt.xyz+opt.dim_e)
            if opt.use_gpu:
                x_in = x_in.cuda()
            e_t = y[0, 0:opt.dim_e]
            # print('e_t', e_t)
            if opt.no_rand:
                if e_t.item() > opt.stop_thresh:
                    x_in[0, -1] = e_t * 0 + 1
                else:
                    x_in[0, -1] = e_t * 0
            else:
                x_in[0, -1] = torch.bernoulli(e_t)
        # print('x_in', x_in)
        if opt.loss_y == 'gmm' or opt.lstm_prop == 'gmm':
            # e_t = y[0, 0:1]
            # pi_t = y[0, 1:21]
            # mu_1_t = y[0, 21:41]
            # mu_2_t = y[0, 41:61]
            # sigma_1_t = y[0, 61:81]
            # sigma_2_t = y[0, 81:101]
            # rho_t = y[0, 101:121]
            pi_t = y[0, opt.dim_e:opt.dim_e + opt.n_component]
            mu_1_t = y[0, opt.dim_e + opt.n_component:opt.dim_e + opt.n_component * 2]
            mu_2_t = y[0, opt.dim_e + opt.n_component * 2:opt.dim_e + opt.n_component * 3]
            rho_t = y[0, opt.dim_e + opt.n_component * 3:opt.dim_e + opt.n_component * 4]
            if not opt.sigma_share:
                sigma_1_t = y[0, opt.dim_e + opt.n_component * 4:opt.dim_e + opt.n_component * 5]
                sigma_2_t = y[0, opt.dim_e + opt.n_component * 5:opt.dim_e + opt.n_component * 6]
            else:
                sigma_1_t = y[0, opt.dim_e + opt.n_component * 4:1 + opt.dim_e + opt.n_component * 4]
                sigma_2_t = y[0, 1 + opt.dim_e + opt.n_component * 4:2 + opt.dim_e + opt.n_component * 4]
                sigma_1_t = sigma_1_t.expand(opt.n_component)
                sigma_2_t = sigma_2_t.expand(opt.n_component)

            # print('='*40, e_t, x_in[0, 2])
            # choice = []
            # for i in range(10):
            #     choice.append()

            _, chosen_pi = torch.max(pi_t, 0)   # value, location
            if i_component is not None:
                pi_t_i = pi_t.detach().clone()
                for i in range(i_component):
                    pi_t_i[chosen_pi] *= 0.
                    _, chosen_pi = torch.max(pi_t_i, 0)
            # random_choice = torch.randint(1, 11, (1,))  # rand int from 1 to 10

            # max_pi = 0
            # for i in range(opt.n_component): # 20
            #     if pi_t[i].item() > max_pi:
            #         max_pi = pi_t[i].item()
            #         index = i

            cur_std = torch.Tensor([sigma_1_t[chosen_pi], sigma_2_t[chosen_pi]])
            cur_cor = torch.Tensor([1, rho_t[chosen_pi]])
            cur_cov_mat = self.make_cov(cur_std, cur_cor)
            cur_mean = torch.Tensor([mu_1_t[chosen_pi], mu_2_t[chosen_pi]])
            ## sample = mvn.rnd()
            # sample = torch.normal(cur_mean, cur_cov_mat)
            # x_in[0, 0] = sample[0]
            # x_in[0, 1] = sample[1]

            x_in[0, 0] = cur_mean[0]
            x_in[0, 1] = cur_mean[1]
        elif opt.loss_y in ['l2', 'l2c', 'l1', 'l1c']:
            x_in[0, :opt.xyz] = y[0, opt.dim_e            :opt.dim_e + 1*opt.xyz]
            x_in[0, opt.xyz:opt.xyz*2] = y[0, opt.dim_e + 1*opt.xyz:opt.dim_e + 2*opt.xyz]
        else:
            raise NotImplementedError

        return x_in

    def make_cov(self, std, rho):
        cov_mat = torch.Tensor(2, 2)
        cov_mat[0, 0] = torch.pow(std[0], 2)
        cov_mat[0, 1] = std[0] * std[1] * rho[1]
        cov_mat[1, 0] = std[0] * std[1] * rho[1]
        cov_mat[1, 1] = torch.pow(std[1], 2)
        return cov_mat

    def forward(self, input_mat, rot_mat, cls_mat=None, depth=None, bbox=None, existence=None,
                prim_box_2d=None, length=None, item_id=None, epoch=-1, c_mask_mat=None, phase=None, length_i=None):
        if opt.encoder == 'resnet' or opt.encoder == 'hg':
            d = self.rgb_encoder(depth)
        elif opt.encoder in ['depth', 'depth_new']:
            d = self.depth_encoder(depth)       # can be shared
        else:
            d = None
        if opt.bbox_con[4:] in ['all', 'bbox']:
            d = torch.cat((d, bbox), dim=1)
        if opt.bbox_con[4:] in ['all', 'exist']:
            d = torch.cat((d, existence), dim=1)
        batch_size = input_mat.size()[0]
        prev_h_0 = collections.OrderedDict({x: torch.zeros(batch_size, opt.hid_size).cuda()
                                            for x in range(opt.hid_layer)})
        prev_c_0 = collections.OrderedDict({x: torch.zeros(batch_size, opt.hid_size).cuda()
                                            for x in range(opt.hid_layer)})
        prev_h = {0: prev_h_0}
        prev_c = {0: prev_c_0}
        if opt.demo is None:
            if opt.proposal == 'load':
                outputs, targets = [], []
                for b in range(item_id.size(0)):
                    output, target = self.train_val_proposal(item_id[b, 0].item(), phase)
                    outputs.append(output)
                    targets.append(target)
            else:
                outputs, targets = self.train_val(input_mat, rot_mat, cls_mat, prim_box_2d, length, item_id,
                                                  d, prev_h, prev_c, epoch, c_mask_mat, phase)
        else:
            if opt.test_loss is not None:
                outputs, targets = self.test_loss(input_mat, rot_mat, cls_mat, prim_box_2d, length, d, prev_h, prev_c)
            else:
                if not opt.gt_in:
                    if opt.proposal == 'load':
                        outputs, targets = self.test_proposal(item_id[0, 0].item(), phase)
                    else:
                        outputs, targets = self.test(input_mat, rot_mat, cls_mat, prim_box_2d, length, d, prev_h, prev_c,
                                                     length_i=length_i)
                else:
                    outputs, targets = self.test_gt(input_mat, rot_mat, cls_mat, prim_box_2d, length, d, prev_h, prev_c)

        return outputs, targets


class PRNNCore(nn.Module):
    def __init__(self):
        super(PRNNCore, self).__init__()
        self.n_sem = opt.n_sem
        if opt.bg_lstm:
            self.n_sem = opt.n_sem + int(opt.len_adjust)
        self.rnn_core = RNNCore()
        if opt.loss_y == 'gmm' or opt.lstm_prop == 'gmm':
            if not opt.sigma_share:
                if opt.sem_reg:
                    self.y_linear = nn.ModuleList([
                        nn.Linear(opt.hid_size * opt.hid_layer, opt.dim_e + opt.n_component * 6)
                    for _ in range(self.n_sem)])
                else:
                    self.y_linear = nn.Linear(opt.hid_size * opt.hid_layer, opt.dim_e + opt.n_component * 6)
            else:
                if opt.sem_reg:
                    self.y_linear = nn.ModuleList([
                        nn.Linear(opt.hid_size * opt.hid_layer, opt.dim_e + 2 + opt.n_component * 4)
                    for _ in range(self.n_sem)])
                else:
                    self.y_linear = nn.Linear(opt.hid_size * opt.hid_layer, opt.dim_e + 2 + opt.n_component * 4)
        elif opt.loss_y in ['l2', 'l2c', 'l1', 'l1c']:
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
        if opt.reg == 's-t-r':
            if opt.sem_reg:
                self.t_linear = nn.ModuleList([nn.Sequential(
                    nn.Linear(opt.hid_size * opt.hid_layer, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 64),  # in_size
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 1 * opt.xyz)
                ) for i in range(self.n_sem)])
            else:
                self.t_linear = nn.Sequential(
                    nn.Linear(opt.hid_size * opt.hid_layer, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 64), # in_size
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, 1 * opt.xyz)
                )
        self.y_hat = YHat()
        if opt.reg != 'str':
            if opt.out_r == '3dprnn':
                self.rot_linear = nn.Sequential(
                    nn.Linear(opt.hid_size * opt.hid_layer, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 64), # in_size
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 32),
                    nn.ReLU(inplace=True)
                )
                self.rot_linear_1 = nn.Sequential(
                    nn.Linear(32, 1*opt.xyz),
                    nn.Tanh()
                )
                self.rot_linear_2 = nn.Sequential(
                    nn.Linear(32, 1*opt.xyz),
                    nn.Sigmoid()
                )
            elif opt.out_r == 'theta':
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
            elif opt.out_r == 'class':
                if opt.sem_reg:
                    self.rot_linear_theta = nn.ModuleList([nn.Sequential(
                        nn.Linear(opt.hid_size * opt.hid_layer, 27),
                        # nn.Linear(opt.hid_size * opt.hid_layer, 256),
                        # nn.ReLU(inplace=True),
                        # nn.Linear(256, 27),
                        nn.Softmax(dim=1)
                    ) for i in range(self.n_sem)])
                    self.rot_linear_axis = nn.ModuleList([nn.Sequential(
                        nn.Linear(opt.hid_size * opt.hid_layer, 4),
                        # nn.Linear(opt.hid_size * opt.hid_layer, 256),
                        # nn.ReLU(inplace=True),
                        # nn.Linear(256, 4),
                        nn.Softmax(dim=1)
                    ) for i in range(self.n_sem)])
                else:
                    self.rot_linear_theta = nn.Sequential(
                        nn.Linear(opt.hid_size * opt.hid_layer, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, 27),
                        nn.Softmax(dim=1)
                    )
                    self.rot_linear_axis = nn.Sequential(
                        nn.Linear(opt.hid_size * opt.hid_layer, 256),
                        nn.ReLU(inplace=True),
                        nn.Linear(256, 4),
                        nn.Softmax(dim=1)
                    )
            else:
                raise NotImplementedError
        if opt.loss_c is not None:
            self.cls_linear = nn.Sequential(
                nn.Linear(opt.hid_size * opt.hid_layer, self.n_sem),
                nn.Softmax(dim=1)
            )
        if opt.loss_box2d is not None:
            if opt.sem_reg and opt.sem_box2d:
                self.box2d_linear = nn.ModuleList([nn.Sequential(
                    nn.Linear(opt.hid_size * opt.hid_layer, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 4)
                ) for i in range(self.n_sem)])
            else:
                self.box2d_linear = nn.Sequential(
                    nn.Linear(opt.hid_size * opt.hid_layer, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, 4)
                )
        if opt.model_name in ['resnet18', 'resnet34']:
            expansion = 1
        elif opt.model_name in ['resnet50', 'resnet101', 'resnet152']:
            expansion = 4
        else:
            raise NotImplementedError
        if 'share' in opt.reg_init:
            self.h_linear = nn.ModuleList([nn.Sequential(
                        nn.Linear(512 * expansion, opt.hid_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(opt.hid_size, opt.hid_size),
                    ) for _ in range(opt.hid_layer)])
            self.c_linear = nn.ModuleList([nn.Sequential(
                        nn.Linear(512 * expansion, opt.hid_size),
                        nn.ReLU(inplace=True),
                        nn.Linear(opt.hid_size, opt.hid_size),
                    ) for _ in range(opt.hid_layer)])

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
                    y.append(self.y_hat(self.y_linear[cls_id](h[ins_i:ins_i + 1])))
            elif out == 'rot':
                y.append(self.rot_linear[cls_id](h[ins_i:ins_i+1]))
            elif out == 'theta':
                y.append(self.rot_linear_theta[cls_id](h[ins_i:ins_i+1]))
            elif out == 'axis':
                y.append(self.rot_linear_axis[cls_id](h[ins_i:ins_i+1]))
            elif out == 'trans':
                y.append(self.t_linear[cls_id](h[ins_i:ins_i+1]))
            elif out == 'box2d':
                y.append(self.box2d_linear[cls_id](h[ins_i:ins_i+1]))
            else:
                raise NotImplementedError
        y = torch.cat(y, dim=0)
        return y

    def forward_by_mask(self, h, cls, out=None, phase=None):
        if out == 'y':
            if opt.reg is not None:
                y_all = [self.y_linear[i](h) for i in range(self.n_sem)]
            else:
                y_all = [self.y_hat(self.y_linear[i](h)) for i in range(self.n_sem)]
        elif out == 'rot':
            y_all = [self.rot_linear[i](h) for i in range(self.n_sem)]
        elif out == 'theta':
            y_all = [self.rot_linear_theta[i](h) for i in range(self.n_sem)]
        elif out == 'axis':
            y_all = [self.rot_linear_axis[i](h) for i in range(self.n_sem)]
        elif out == 'trans':
            y_all = [self.t_linear[i](h) for i in range(self.n_sem)]
        elif out == 'box2d':
            y_all = [self.box2d_linear[i](h) for i in range(self.n_sem)]
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

    def forward(self, x, r, cls=None, box2d=None, d=None, local=None, img_f=None,
                prev_h=None, prev_c=None, step=None, targets=None, phase=None):
        # pdb.set_trace()
        if step == 0 and 'share' in opt.reg_init:
            next_h = collections.OrderedDict({l: self.h_linear[l](img_f) for l in range(opt.hid_layer)})
            next_c = collections.OrderedDict({l: self.c_linear[l](img_f) for l in range(opt.hid_layer)})
        else:
            next_h, next_c = self.rnn_core(x, r, cls, box2d, d, prev_h, prev_c)
        h = torch.cat(tuple(next_h[l] for l in range(opt.hid_layer)), dim=1)
        cls_update = cls
        if opt.loss_c is not None:
            if (step + opt.step_start) % (3 // opt.xyz) == 0:
                cls_update = self.cls_linear(h)
            if opt.sem_select == 'gt' and phase == 'train' and opt.refine == None and opt.bi_lstm == None:
                cls = targets[step][2]
            else:
                cls = cls_update
        if opt.loss_box2d is not None:
            if (step + opt.step_start) % (3 // opt.xyz) == 0:
                if opt.sem_reg and opt.sem_box2d:
                    box2d = self.forward_by_ins(h, cls, out='box2d', phase=phase)
                else:
                    box2d = self.box2d_linear(h)
        if opt.sem_reg:
            y = self.forward_by_ins(h, cls, out='y', phase=phase)
        else:
            y = self.y_hat(self.y_linear(h))
        if opt.reg == 's-t-r':
            if opt.sem_reg:
                t = self.forward_by_ins(h, cls, out='trans', phase=phase)
            else:
                t = self.t_linear(h)
            y = torch.cat((y, t), dim=1)
        if opt.reg == 'str':
            rot = y[:, 2 * opt.xyz:]
            y = y[:, :2 * opt.xyz]
        else:
            if opt.out_r == '3dprnn':
                rot_branch = self.rot_linear(h)
                rot_1 = self.rot_linear_1(rot_branch)
                rot_2 = self.rot_linear_2(rot_branch)
                rot = torch.cat((rot_1, rot_2), dim=1)
            elif opt.out_r == 'theta':
                if opt.sem_reg:
                    rot = self.forward_by_ins(h, cls, out='rot', phase=phase)
                else:
                    rot = self.rot_linear(h)
            elif opt.out_r == 'class':
                if opt.sem_reg:
                    theta = self.forward_by_ins(h, cls, out='theta', phase=phase)
                    axis = self.forward_by_ins(h, cls, out='axis', phase=phase)
                    rot = torch.cat((theta, axis), dim=1)
                else:
                    theta = self.rot_linear_theta(h)
                    axis = self.rot_linear_axis(h)
                    rot = torch.cat((theta, axis), dim=1)
            else:
                raise NotImplementedError
        return y, rot, cls_update, box2d, next_h, next_c


class RNNCore(nn.Module):
    def __init__(self):
        super(RNNCore, self).__init__()
        self.lstms = nn.ModuleList([LSTMUnit(layer_id=i) for i in range(opt.hid_layer)])

    def forward(self, x, r, cls=None, box2d=None, d=None, prev_h=None, prev_c=None):
        next_h = collections.OrderedDict()
        next_c = collections.OrderedDict()
        for l in range(opt.hid_layer):
            if l > 0:
                below_h = next_h[l - 1]
            else:
                below_h = None
            next_h[l], next_c[l] = self.lstms[l](x, r, cls, box2d, d, prev_h[l], prev_c[l], below_h)

        return next_h, next_c


class LSTMUnit(nn.Module):
    def __init__(self, layer_id=0):
        super(LSTMUnit, self).__init__()
        if opt.out_r == '3dprnn':
            out_size = 4*opt.xyz + opt.dim_e
        elif opt.out_r == 'theta':
            out_size = 3*opt.xyz + opt.dim_e
        elif opt.out_r == 'class':
            out_size = 2*opt.xyz + 31 + opt.dim_e
        else:
            raise NotImplementedError
        if opt.loss_c is not None and opt.loss_box2d is not None:
            out_size = out_size + opt.n_sem + 4
        elif opt.loss_c is not None:
            out_size = out_size + opt.n_sem
        elif opt.loss_box2d is not None:
            out_size = out_size + 4
        else:
            out_size = out_size
        self.i2hs = nn.ModuleList([nn.Linear(out_size, opt.hid_size) for i in range(4)])
        self.h2hs = nn.ModuleList([nn.Linear(opt.hid_size, opt.hid_size) for i in range(4)])
        if layer_id > 0:
            self.bh2hs = nn.ModuleList([nn.Linear(opt.hid_size, opt.hid_size) for i in range(4)])
        # if opt.depth_con or opt.rgb_con:
        if opt.encoder is not None:
            c_size = opt.con_size
            if opt.box2d_pos == '0' and opt.box2d_size > 0:
                c_size += opt.box2d_size
                if opt.box2d_en == 'hard':
                    c_size += 4
            if opt.bbox_con[4:] in ['all', 'bbox']:
                c_size += opt.n_sem * opt.n_para
            if opt.bbox_con[4:] in ['all', 'exist']:
                c_size += opt.n_sem
            self.w2hs = nn.ModuleList([nn.Linear(c_size, opt.hid_size) for i in range(4)])
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.layer_id = layer_id

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(opt.hid_size)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    def forward(self, x, r, cls=None, box2d=None, d=None, prev_h=None, prev_c=None, below_h=None):
        if opt.loss_c is not None and opt.loss_box2d is not None:
            inputs = torch.cat((x, r, cls, box2d), dim=1)
        elif opt.loss_c is not None:
            inputs = torch.cat((x, r, cls), dim=1)
        elif opt.loss_box2d is not None:
            # print(x)
            # print(r)
            # print(box2d)
            inputs = torch.cat((x, r, box2d), dim=1)
        else:
            inputs = torch.cat((x, r), dim=1)
        gates = {}  # in_gate, forget_gate, out_gate, in_transform
        for i in range(4):
            # print('inputs', inputs.size())
            gates[i] = self.i2hs[i](inputs) + self.h2hs[i](prev_h)
            if self.layer_id > 0:
                gates[i] += self.bh2hs[i](below_h)
            # if opt.depth_con or opt.rgb_con:
            if opt.encoder is not None:
                gates[i] += self.w2hs[i](d)
            if i <= 2:
                gates[i] = self.sigmoid(gates[i])
            else:
                gates[i] = self.tanh(gates[i])
        next_c = gates[1] * prev_c + gates[0] * gates[3]
        next_h = gates[2] * self.tanh(next_c)

        return next_h, next_c


class YHat(nn.Module):
    def __init__(self):
        super(YHat, self).__init__()
        if opt.loss_y == 'gmm' or opt.lstm_prop == 'gmm':
            self.e_t_act = nn.Sigmoid()
            self.pi_t_act = nn.Softmax(dim=1)
            self.rho_t_act = nn.Tanh()
        elif opt.loss_y in ['l2', 'l2c', 'l1', 'l1c']:
            self.e_t_act = nn.Sigmoid()
            # self.st_act = nn.ReLU(inplace=True)

    def count_if_sigma_is_too_small(self, sigma):
        count = torch.sum(sigma < 0.0001)
        opt.c_sigma += count.item()
        if count.item() > 0:
            print('=' * 50, 'count: ', count.item(), 'c_sigma: ', opt.c_sigma)
            print(sigma)

    def forward(self, hat_h):
        if opt.loss_e is not None:
            hat_e_t = hat_h[:, 0:opt.dim_e]
            e_t = self.e_t_act(-hat_e_t)
        else:
            e_t = None

        if opt.loss_y == 'gmm' or opt.lstm_prop == 'gmm':
            # hat_e_t = hat_h[:, 0:1]
            # hat_pi_t = hat_h[:, 1:21]
            # hat_mu_1_t = hat_h[:, 21:41]
            # hat_mu_2_t = hat_h[:, 41:61]
            # hat_sigma_1_t = hat_h[:, 61:81]
            # hat_sigma_2_t = hat_h[:, 81:101]
            # hat_rho_t = hat_h[:, 101:121]
            hat_pi_t = hat_h[:, opt.dim_e:opt.dim_e + opt.n_component]
            hat_mu_1_t = hat_h[:, opt.dim_e + opt.n_component:opt.dim_e + opt.n_component * 2]
            hat_mu_2_t = hat_h[:, opt.dim_e + opt.n_component * 2:opt.dim_e + opt.n_component * 3]
            hat_rho_t = hat_h[:, opt.dim_e + opt.n_component * 3:opt.dim_e + opt.n_component * 4]
            if not opt.sigma_share:
                hat_sigma_1_t = hat_h[:, opt.dim_e + opt.n_component * 4:opt.dim_e + opt.n_component * 5]
                hat_sigma_2_t = hat_h[:, opt.dim_e + opt.n_component * 5:opt.dim_e + opt.n_component * 6]
            else:
                hat_sigma_1_t = hat_h[:, opt.dim_e + opt.n_component * 4:opt.dim_e + 1 + opt.n_component * 4]
                hat_sigma_2_t = hat_h[:, opt.dim_e + 1 + opt.n_component * 4:opt.dim_e + 2 + opt.n_component * 4]

            pi_t = self.pi_t_act(hat_pi_t)
            mu_1_t = hat_mu_1_t
            mu_2_t = hat_mu_2_t
            if not opt.sigma_abs:
                sigma_1_t = torch.exp(hat_sigma_1_t) + opt.sigma_min
                sigma_2_t = torch.exp(hat_sigma_2_t) + opt.sigma_min
            else:
                sigma_1_t = self.e_t_act(hat_sigma_1_t) + opt.sigma_min
                sigma_2_t = self.e_t_act(hat_sigma_2_t) + opt.sigma_min
            rho_t = self.rho_t_act(hat_rho_t)

            self.count_if_sigma_is_too_small(sigma_1_t)
            self.count_if_sigma_is_too_small(sigma_2_t)

            # print(e_t, pi_t, mu_1_t, mu_2_t, sigma_1_t, sigma_2_t, rho_t)
            if opt.loss_e is not None:
                y = torch.cat((e_t, pi_t, mu_1_t, mu_2_t, rho_t, sigma_1_t, sigma_2_t), dim=1)
            else:
                y = torch.cat((pi_t, mu_1_t, mu_2_t, rho_t, sigma_1_t, sigma_2_t), dim=1)
        elif opt.loss_y in ['l2', 'l2c', 'l1', 'l1c']:
            hat_s_t = hat_h[:, opt.dim_e            :opt.dim_e + 1*opt.xyz]
            hat_t_t = hat_h[:, opt.dim_e + 1*opt.xyz:opt.dim_e + 2*opt.xyz]
            # s_t = self.st_act(hat_s_t)
            # t_t = self.st_act(hat_t_t)
            s_t = hat_s_t
            t_t = hat_t_t
            if opt.loss_e is not None:
                y = torch.cat((e_t, s_t, t_t), dim=1)
            else:
                y = torch.cat((s_t, t_t), dim=1)
        else:
            raise NotImplementedError

        return y


class VOXAEnet(nn.Module):
    def __init__(self, out_size=None):
        super(VOXAEnet, self).__init__()
        global opt
        opt = get_opt()
        if out_size is None:
            out_size = opt.con_size
        self.conv1 = nn.Conv2d(1, 32, 7, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, out_size)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.pool(self.relu(self.conv2(out)))
        out = self.pool(self.relu(self.conv3(out)))
        out = self.linear1(out.view(out.size()[0], -1))
        out = self.linear2(out)

        return out


class MaskCriterion(nn.Module):
    def __init__(self, metric=None):
        super(MaskCriterion, self).__init__()
        global opt
        opt = get_opt()
        self.metric = metric
        self.mask = None
        self.size_average = False
        if metric == 'l2c':
            self.criterion = nn.MSELoss()
        elif metric == 'l1c':
            self.criterion = nn.L1Loss()
        elif metric == 'bcec':
            self.criterion = nn.BCELoss()
        elif metric == 'sl1c':
            self.criterion = nn.SmoothL1Loss()
        elif metric == 'nllc':
            self.criterion = nn.NLLLoss()
        else:
            raise NotImplementedError
        if opt.use_gpu:
            self.criterion =  self.criterion.cuda()

    def set_mask(self, mask):
        self.mask = mask

    def set_size_average(self):
        self.size_average = True

    def forward(self, inputs, targets):
        sample_size = inputs.size()[0]
        loss = torch.zeros(1)[0]
        if opt.use_gpu:
            loss = loss.cuda()
        # print('inputs', inputs)
        # print('targets', targets)
        # print('mask', self.mask)
        for i in range(sample_size):
            if self.mask[i, 0].item() > 0:
                loss_i = self.criterion(inputs[i:i+1], targets[i:i+1]) * self.mask[i, 0]
                loss += loss_i
            # print(i, loss_i, self.mask[i, 0], loss)
        # print('count', torch.sum(self.mask))
        if self.size_average:
            loss /= torch.sum(self.mask)
        # print('loss', loss)
        return loss


class MixtureCriterion(nn.Module):
    def __init__(self):
        super(MixtureCriterion, self).__init__()
        self.mask = None
        self.size_average = False

    def set_mask(self, mask):
        self.mask = mask

    def set_size_average(self):
        self.size_average = True

    def forward(self, inputs, targets):
        sample_size = inputs.size()[0]
        x1 = targets[:, 0:1*opt.xyz]
        x2 = targets[:, 1*opt.xyz:2*opt.xyz]
        if opt.loss_e is not None:
            x3 = targets[:, 2*opt.xyz:]
            e_t = inputs[:, 0:opt.dim_e]
            if opt.use_gpu:
                eq1 = torch.eq(x3, torch.ones(sample_size, 1).cuda()).float()
                neq1 = torch.ne(x3, torch.ones(sample_size, 1).cuda()).float()
            else:
                eq1 = torch.eq(x3, torch.ones(sample_size, 1)).float()
                neq1 = torch.ne(x3, torch.ones(sample_size, 1)).float()
            eq1 *= torch.log(e_t + 1e-15)
            neq1 *= torch.log(-e_t + 1 + 1e-15)
            log_e_t = torch.neg(eq1 + neq1)
            log_e_t *= self.mask
            log_e_t = torch.sum(log_e_t)
            if self.size_average:
                log_e_t /= targets.size()[0]
        else:
            log_e_t = 0

        if opt.loss_y == 'gmm':
            pi_t = inputs[:, opt.dim_e:opt.dim_e + opt.n_component]
            mu_1_t = inputs[:, opt.dim_e + opt.n_component:opt.dim_e + opt.n_component * 2]
            mu_2_t = inputs[:, opt.dim_e + opt.n_component * 2:opt.dim_e + opt.n_component * 3]
            rho_t = inputs[:, opt.dim_e + opt.n_component * 3:opt.dim_e + opt.n_component * 4]
            if not opt.sigma_share:
                sigma_1_t = inputs[:, opt.dim_e + opt.n_component * 4:opt.dim_e + opt.n_component * 5]
                sigma_2_t = inputs[:, opt.dim_e + opt.n_component * 5:opt.dim_e + opt.n_component * 6]
            else:
                sigma_1_t = inputs[:, opt.dim_e + opt.n_component * 4:opt.dim_e + 1 + opt.n_component * 4]
                sigma_2_t = inputs[:, opt.dim_e + 1 + opt.n_component * 4:opt.dim_e + 2 + opt.n_component * 4]
                sigma_1_t = sigma_1_t.expand(sample_size, opt.n_component)
                sigma_2_t = sigma_2_t.expand(sample_size, opt.n_component)

            inv_sigma1 = torch.pow(sigma_1_t + 1e-15, -1)
            inv_sigma2 = torch.pow(sigma_2_t + 1e-15, -1)

            mixdist1 = inv_sigma1 * inv_sigma2
            mixdist1 *= torch.pow(-torch.pow(rho_t, 2) + 1 + 1e-15, -0.5)
            mixdist1 *= 1 / (2 * math.pi)

            mu_1_x_1 = torch.neg(mu_1_t)
            mu_2_x_2 = torch.neg(mu_2_t)

            x1_val = x1.expand(sample_size, opt.n_component) # 20
            x2_val = x2.expand(sample_size, opt.n_component) # 20
            mu_1_x_1 += x1_val
            mu_2_x_2 += x2_val

            mixdist2_z_1 = torch.pow(inv_sigma1, 2) * torch.pow(mu_1_x_1, 2)
            mixdist2_z_2 = torch.pow(inv_sigma2, 2) * torch.pow(mu_2_x_2, 2)
            mixdist2_z_3 = inv_sigma1 * inv_sigma2

            mixdist2_z_3 *= mu_1_x_1
            mixdist2_z_3 *= mu_2_x_2
            # print(mixdist2_z_3)

            mixdist2_z_3 *= rho_t * 2
            mixdist2 = mixdist2_z_1 + mixdist2_z_2 - mixdist2_z_3
            mixdist2 = torch.neg(mixdist2)
            mixdist2 *= torch.pow((-torch.pow(rho_t, 2) + 1 + 1e-15) * 2, -1)
            mixdist2 = torch.exp(mixdist2)
            mixdist = mixdist1 * mixdist2
            mixdist *= pi_t
            # print(mixdist)

            mixdist_sum = torch.sum(mixdist, 1, keepdim=True)
            mixdist_sum += 1e-15

            log_mixdist_sum = torch.log(mixdist_sum)
            # print(log_mixdist_sum)
            loss_y = torch.neg(log_mixdist_sum)
            # result = loss_y + log_e_t
            loss_y *= self.mask
            loss_y = torch.sum(loss_y)
            if opt.sigma_reg > 0.:
                loss_s = -opt.sigma_reg * (torch.sum(torch.abs(sigma_1_t)) +
                                           torch.sum(torch.abs(sigma_2_t))) / opt.n_component / 2.
            else:
                loss_s = torch.zeros(1)[0]
                if opt.use_gpu:
                    loss_s = loss_s.cuda()
            if self.size_average:
                loss_y /= targets.size()[0]
                loss_s /= targets.size()[0]
            return loss_y, log_e_t, loss_s
        else:
            return log_e_t


class BBoxCriterion(nn.Module):
    def __init__(self):
        super(BBoxCriterion, self).__init__()
        self.mask = None
        self.size_average = False

    def set_mask(self, mask):
        self.mask = mask

    def set_size_average(self):
        self.size_average = True

    def forward(self, pred_out, bbox, existence):
        corners = self.get_corners(pred_out)

    def get_corners(self, pred_out):
        pass

    def getVx(self, Rv):
        vx = np.array([[0, -Rv[2], Rv[1]],
            [Rv[2], 0, -Rv[0]],
            [-Rv[1], Rv[0], 0]])
        return vx

    def get_mean(self, shape,trans,scale,Rrot):
        #cen_mean = [14.343102,22.324961,23.012661]
        cen_mean = [trans[0]+shape[0]/2, trans[1]+shape[1]/2, trans[2]+shape[2]/2]
        # print(cen_mean)
        return cen_mean

    def get_sym(self, prim_r, voxel_scale):
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


def mix_loss(mask, inputs, targets):
    x1 = targets[:, 0:1]
    x2 = targets[:, 1:2]
    x3 = targets[:, 2:3]
    # print('x1, x2, x3')
    # print(x1, x2, x3)
    sample_size = inputs.size()[0]

    # e_t = inputs[:, 0:1]
    # pi_t = inputs[:, 1:21]
    # mu_1_t = inputs[:, 21:41]
    # mu_2_t = inputs[:, 41:61]
    # sigma_1_t = inputs[:, 61:81]
    # sigma_2_t = inputs[:, 81:101]
    # rho_t = inputs[:, 101:121]
    e_t = inputs[:, 0:1]
    pi_t = inputs[:, 1:1 + opt.n_component]
    mu_1_t = inputs[:, 1 + opt.n_component:1 + opt.n_component * 2]
    mu_2_t = inputs[:, 1 + opt.n_component * 2:1 + opt.n_component * 3]
    rho_t = inputs[:, 1 + opt.n_component * 3:1 + opt.n_component * 4]
    if not opt.sigma_share:
        sigma_1_t = inputs[:, 1 + opt.n_component * 4:1 + opt.n_component * 5]
        sigma_2_t = inputs[:, 1 + opt.n_component * 5:1 + opt.n_component * 6]
    else:
        sigma_1_t = inputs[:, 1 + opt.n_component * 4:2 + opt.n_component * 4]
        sigma_2_t = inputs[:, 2 + opt.n_component * 4:3 + opt.n_component * 4]
        sigma_1_t = sigma_1_t.expand(sample_size, opt.n_component)
        sigma_2_t = sigma_2_t.expand(sample_size, opt.n_component)


    # print(sample_size)

    inv_sigma1 = torch.pow(sigma_1_t + 1e-15, -1)
    inv_sigma2 = torch.pow(sigma_2_t + 1e-15, -1)
    # print('inv_sigma')
    # print(inv_sigma1, inv_sigma2)

    mixdist1 = inv_sigma1 * inv_sigma2
    mixdist1 *= torch.pow(-torch.pow(rho_t, 2) + 1 + 1e-15, -0.5)
    mixdist1 *= 1 / (2 * math.pi)
    # print(mixdist1)

    mu_1_x_1 = torch.neg(mu_1_t)
    mu_2_x_2 = torch.neg(mu_2_t)
    # print('mu_1_x_1')
    # print(mu_1_x_1)
    # print(mu_2_x_2)

    x1_val = x1.expand(sample_size, opt.n_component) # 20
    x2_val = x2.expand(sample_size, opt.n_component) # 20
    # print('x1_val')
    # print(x1_val, x2_val)
    mu_1_x_1 += x1_val
    mu_2_x_2 += x2_val
    # print('mu_1_x_1')
    # print(mu_1_x_1)
    # print(mu_2_x_2)

    mixdist2_z_1 = torch.pow(inv_sigma1, 2) * torch.pow(mu_1_x_1, 2)
    mixdist2_z_2 = torch.pow(inv_sigma2, 2) * torch.pow(mu_2_x_2, 2)
    mixdist2_z_3 = inv_sigma1 * inv_sigma2
    # print('mixdist2')
    # print(mixdist2_z_1)
    # print(mixdist2_z_2)
    # print(mixdist2_z_3)

    mixdist2_z_3 *= mu_1_x_1
    mixdist2_z_3 *= mu_2_x_2
    # print(mixdist2_z_3)

    mixdist2_z_3 *= rho_t * 2
    mixdist2 = mixdist2_z_1 + mixdist2_z_2 - mixdist2_z_3
    mixdist2 = torch.neg(mixdist2)
    mixdist2 *= torch.pow((-torch.pow(rho_t, 2) + 1 + 1e-15) * 2, -1)
    mixdist2 = torch.exp(mixdist2)
    mixdist = mixdist1 * mixdist2
    mixdist *= pi_t
    # print(mixdist)

    mixdist_sum = torch.sum(mixdist, 1, keepdim=True)
    mixdist_sum += 1e-15
    # print(mixdist_sum)

    log_mixdist_sum = torch.log(mixdist_sum)
    # print(log_mixdist_sum)

    if opt.use_gpu:
        eq1 = torch.eq(x3, torch.ones(sample_size, 1).cuda()).float()
        neq1 = torch.ne(x3, torch.ones(sample_size, 1).cuda()).float()
    else:
        eq1 = torch.eq(x3, torch.ones(sample_size, 1)).float()
        neq1 = torch.ne(x3, torch.ones(sample_size, 1)).float()
    eq1 *= torch.log(e_t + 1e-15)
    neq1 *= torch.log(-e_t + 1 + 1e-15)
    log_e_t = eq1 + neq1
    # print(log_e_t)

    result = log_mixdist_sum + log_e_t
    # print(result)
    result = torch.neg(result)
    # print(result)
    result *= mask
    # print(result)
    result = torch.sum(result)
    # print(result)
    # print(targets.size())
    # print(result)
    if True:
        result /= targets.size()[0]
    # print(result)

    return result
