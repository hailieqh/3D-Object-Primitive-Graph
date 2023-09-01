import collections
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.opts import *
from lib.utils.prnn_utils import *
import pdb


class GraphNet(nn.Module):
    def __init__(self, node, embed_type):
        super(GraphNet, self).__init__()
        global opt
        opt = get_opt()
        self.xyz = opt.graph_xyz
        self.node = node
        g1_in_size = get_g1_in_size(self.node)
        # self.stacks = nn.ModuleList([GraphStack() for _ in range(opt.n_sem)])
        self.graph_stack = nn.ModuleList([GraphStack(self.node, embed_type, g1_in_size, self.xyz) for _ in range(opt.stack_mp)])
        self.embed_stack = nn.ModuleList([nn.Linear(opt.f_dim, g1_in_size) for _ in range(opt.stack_mp - 1)])
        if opt.loss_y in ['l2', 'l2c', 'l1', 'l1c']:
            if opt.reg == 'str':
                out_size = 2 * self.xyz + opt.dim_e + 1 * self.xyz
            elif opt.reg == 's-t-r':
                out_size = 1 * self.xyz + opt.dim_e
            else:
                out_size = 2 * self.xyz + opt.dim_e
            if opt.sem_graph:
                self.y_linear = nn.ModuleList([nn.Sequential(
                    nn.Linear(opt.f_dim, opt.m_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(opt.m_dim, 64),  # in_size
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, out_size)
                ) for i in range(opt.n_sem + int(opt.len_adjust))])
            else:
                self.y_linear = nn.Sequential(
                    nn.Linear(opt.f_dim, opt.m_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(opt.m_dim, 64),  # in_size
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, out_size)
                )
        if opt.out_r == '3dprnn':
            self.rot_linear = nn.Sequential(
                nn.Linear(opt.f_dim, opt.m_dim),
                nn.ReLU(inplace=True),
                nn.Linear(opt.m_dim, 64),  # in_size
                nn.ReLU(inplace=True),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True)
            )
            self.rot_linear_1 = nn.Sequential(
                nn.Linear(32, 1 * self.xyz),
                nn.Tanh()
            )
            self.rot_linear_2 = nn.Sequential(
                nn.Linear(32, 1 * self.xyz),
                nn.Sigmoid()
            )
        elif opt.out_r == 'theta':
            if opt.sem_graph:
                self.rot_linear = nn.ModuleList([nn.Sequential(
                    nn.Linear(opt.f_dim, opt.m_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(opt.m_dim, 1 * self.xyz)
                ) for i in range(opt.n_sem + int(opt.len_adjust))])
            else:
                self.rot_linear = nn.Sequential(
                    nn.Linear(opt.f_dim, opt.m_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(opt.m_dim, 1 * self.xyz)
                )
        if opt.loss_c is not None or opt.len_adjust:
            self.cls_linear = nn.Sequential(
                nn.Linear(opt.f_dim, opt.n_sem + int(opt.len_adjust)),
                nn.Softmax(dim=1)
            )

    def cls_prob_to_one_hot(self, cls):
        _, cls_id = torch.max(cls, dim=1)
        # cls_one_hot = torch.zeros(cls.size())
        cls_one_hot = torch.zeros((cls.size(0), opt.n_sem + int(opt.len_adjust)))
        cls_one_hot[0, cls_id[0]] = 1
        cls_one_hot = cls_one_hot.cuda()
        return cls_id, cls_one_hot

    def batch_inputs_to_node(self, b, node_i, inputs, prev_h=None, features=None, targets=None):
        y, rot, cls, box2d = inputs[node_i]
        y, rot = y[b:b + 1], rot[b:b + 1]
        if opt.loss_box2d is not None:
            box2d = box2d[b:b + 1]
        cls = cls[b:b + 1]
        # assert opt.loss_c is not None
        cls_id, cls_one_hot = self.cls_prob_to_one_hot(cls)
        node = collections.OrderedDict()
        if 'img' in self.node or 'img' in opt.partial_graph:
            img = features[node_i][b:b + 1]
            node['img'] = img
        if 'h' in self.node or 'h' in opt.partial_graph:
            if 'proposal' in opt.faster:
                h = prev_h[node_i][b:b + 1]
            else:
                h = torch.cat(tuple(prev_h[node_i + 1][l][b:b + 1] for l in range(opt.hid_layer)), dim=1)
            node['h'] = h
        if 'n' in self.node or 'n' in opt.partial_graph:
            node['n'] = prev_h[b][node_i]
        if 'st' in self.node or 'st' in opt.partial_graph:
            node['st'] = y
        if 'r' in self.node or 'r' in opt.partial_graph:
            node['r'] = rot
        if 'cls' in self.node or 'cls' in opt.partial_graph:
            if opt.len_adjust and cls.size(1) == opt.n_sem:
                cls = torch.cat((cls, torch.zeros(cls.size(0), 1).cuda()), dim=1)
            node['cls'] = cls
        if 'cot' in self.node or 'cot' in opt.partial_graph:
            node['cot'] = cls_one_hot
        # node = torch.cat(node, dim=1)
        return node, cls_id, cls_one_hot

    def get_batch_graph_init(self, b, inputs, length, prev_h=None, features=None, targets=None):
        nodes = []
        nodes_sem_labels = []
        for node_i in range(length):
            node, cls_id, cls_one_hot = self.batch_inputs_to_node(b, node_i, inputs, prev_h=prev_h,
                                                                  features=features, targets=targets)
            nodes.append((node, cls_one_hot))
            nodes_sem_labels.append((cls_id[0].item(), cls_one_hot))
        return nodes, nodes_sem_labels

    def get_batch_graph_stack_mp(self, p, length, nodes, nodes_sem_labels):
        nodes_out = []
        for node_i in range(length):
            node = self.embed_stack[p - 1](nodes[node_i])
            cls_one_hot = nodes_sem_labels[node_i][1]
            nodes_out.append((node, cls_one_hot))
        return nodes_out

    def mp_batch_graph(self, p, nodes):
        nodes = self.graph_stack[p](nodes)
        return nodes

    def update_outputs_batch(self, length, nodes, nodes_sem_labels, seq_all, out=None):
        seq = [0 for _ in range(length)]  # 11
        nodes_sem_labels_new = []
        for i in range(length):
            if out == 'y':
                sem_i = nodes_sem_labels[i][0]
                if opt.sem_graph:
                    output = self.y_linear[sem_i](nodes[i])
                else:
                    output = self.y_linear(nodes[i])
            elif out == 'rot':
                sem_i = nodes_sem_labels[i][0]
                if opt.sem_graph:
                    output = self.rot_linear[sem_i](nodes[i])
                else:
                    output = self.rot_linear(nodes[i])
            elif out == 'cls':
                output = self.cls_linear(nodes[i])
                cls_id, cls_one_hot = self.cls_prob_to_one_hot(output)
                nodes_sem_labels_new.append((cls_id[0].item(), cls_one_hot))
            else:
                raise NotImplementedError
            seq[i] = output  # 1, 6
        seq_all.append(seq)
        return seq_all, nodes_sem_labels_new

    def go_through_graph(self, lengths, inputs, prev_h=None, features=None, targets=None):
        # lengths = [12, 15, 9, ..., 9] # len 16 (batch_size)
        cls_seq_all, y_seq_all, rot_seq_all = [], [], []
        nodes_final = []
        for b in range(len(lengths)):
            length = get_batch_length(b, lengths)
            nodes_all, nodes_sem_labels_all = [], []
            cls_seq_all_p = []
            for p in range(opt.stack_mp):
                if p == 0:  # initial cls
                    nodes, nodes_sem_labels = self.get_batch_graph_init(b, inputs, length, prev_h=prev_h,
                                                                        features=features, targets=targets)
                else:
                    nodes = nodes_all[p - 1]
                    nodes_sem_labels = nodes_sem_labels_all[p - 1]
                    pdb.set_trace()
                    nodes = self.get_batch_graph_stack_mp(p, length, nodes, nodes_sem_labels)
                nodes = self.mp_batch_graph(p, nodes)
                if 'cls' in opt.update:  # update cls
                    cls_seq_all_p, nodes_sem_labels = self.update_outputs_batch(length, nodes, nodes_sem_labels,
                                                                                cls_seq_all_p, out='cls')
                nodes_all.append(nodes)
                nodes_sem_labels_all.append(nodes_sem_labels)
                if p == opt.stack_mp - 1:
                    if 'cls' in opt.update:
                        cls_seq_all.append(cls_seq_all_p[-1])
                    y_seq_all, _ = self.update_outputs_batch(length, nodes, nodes_sem_labels, y_seq_all, out='y')
                    rot_seq_all, _ = self.update_outputs_batch(length, nodes, nodes_sem_labels, rot_seq_all, out='rot')
            nodes_final.append(nodes_all[-1])
        cls_outs = []
        if 'cls' in opt.update:
            cls_outs = out_seq_to_batch(lengths, cls_seq_all)
        y_outs = out_seq_to_batch(lengths, y_seq_all)
        rot_outs = out_seq_to_batch(lengths, rot_seq_all)
        return nodes_final, y_outs, rot_outs, cls_outs

    def forward(self, lengths, inputs, prev_h=None, features=None, targets=None, phase=None):
        # lengths = [12, 15, 9, ..., 9] # len 16 (batch_size)
        nodes_final, y_outs, rot_outs, cls_outs = self.go_through_graph(lengths, inputs, prev_h, features, targets)
        cls_update = 'cls' in opt.update
        outputs = wrap_up_outputs(self.xyz, lengths, inputs, y_outs, rot_outs,
                                  cls_outs=cls_outs, cls_update=cls_update, phase=phase)
        return outputs, nodes_final


def get_batch_length(b, lengths):
    length = lengths[b] - 1
    if opt.reg_init != 'None':
        length = lengths[b]
    return length


def out_seq_to_batch(lengths, out_seq_all):
    eps = 1e-8
    max_len = max(lengths) - 1  # maybe 33
    if opt.reg_init != 'None':
        max_len = max(lengths)
    out_batch = [torch.zeros(out_seq_all[0][0].size()).cuda()+eps for x in range(max_len)]  # (1, 256) *33
    for node_i in range(max_len):  # 0-31
        for b in range(len(lengths)):  # 0-15
            length = get_batch_length(b, lengths)
            if node_i < length:  # < 12
                if out_seq_all[b][node_i] is None:
                    out_seq_all[b][node_i] = torch.zeros(out_seq_all[0][0].size()).cuda() + eps
                if b == 0:
                    out_batch[node_i] = out_seq_all[b][node_i]
                else:
                    out_batch[node_i] = torch.cat((out_batch[node_i],
                                                   out_seq_all[b][node_i]), dim=0)
            else:
                if b > 0:
                    out_batch[node_i] = torch.cat((out_batch[node_i],
                                                   torch.zeros(out_seq_all[0][0].size()).cuda() + eps), dim=0)
    return out_batch


def wrap_up_outputs(xyz, lengths, inputs, y_outs, rot_outs, cls_outs=None, cls_update=False, phase=None):
    outputs = []
    max_len = max(lengths) - 1  # maybe 10
    if opt.reg_init != 'None':
        max_len = max(lengths)
    if phase == 'train':
        batch_sze = opt.train_batch
    elif phase == 'val':## incorrect assignment but no harm
        batch_sze = opt.valid_batch
    else:
        raise NotImplementedError
    for i in range(len(inputs)):
        _, _, cls, box2d = inputs[i]
        if i < max_len:
            y, rot = y_outs[i], rot_outs[i]
            if cls_update:
                cls = cls_outs[i]
        else:
            # y = torch.zeros(y_outs[0].size()).cuda()
            # rot = torch.zeros(rot_outs[0].size()).cuda()
            y = torch.zeros(batch_sze, xyz * 2).cuda()
            rot = torch.zeros(batch_sze, xyz).cuda()
        outputs.append((y, rot, cls, box2d))
    return outputs


class PrimMLPs(nn.Module):
    def __init__(self):
        super(PrimMLPs, self).__init__()
        global opt
        opt = get_opt()
        self.xyz = opt.graph_xyz
        if opt.loss_y in ['l2', 'l2c', 'l1', 'l1c']:
            if opt.reg == 'str':
                out_size = 2 * self.xyz + opt.dim_e + 1 * self.xyz
            elif opt.reg == 's-t-r':
                out_size = 1 * self.xyz + opt.dim_e
            else:
                out_size = 2 * self.xyz + opt.dim_e
            self.y_linear_final = nn.ModuleList([nn.Sequential(
                nn.Linear(opt.f_dim, opt.m_dim),
                nn.ReLU(inplace=True),
                nn.Linear(opt.m_dim, 64),  # in_size
                nn.ReLU(inplace=True),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, out_size)
            ) for i in range(opt.n_sem + int(opt.len_adjust))])
        if opt.out_r == 'theta':
            self.rot_linear_final = nn.ModuleList([nn.Sequential(
                nn.Linear(opt.f_dim, opt.m_dim),
                nn.ReLU(inplace=True),
                nn.Linear(opt.m_dim, 1 * self.xyz)
            ) for i in range(opt.n_sem + int(opt.len_adjust))])

    def forward(self, lengths, inputs, nodes_all, cls_gt_all, phase=None):
        cls_seq_all, y_seq_all, rot_seq_all = [], [], []
        for b in range(len(lengths)):
            length = get_batch_length(b, lengths)
            if opt.sem_select == 'gt' and phase == 'train':
                nodes_sem_labels = cls_gt_all[b]
            else:
                nodes_sem_labels = [inputs[i][2][b:b+1] for i in range(length)]
            y_seq_all = self.update_outputs_batch(length, nodes_all[b], nodes_sem_labels,
                                                  y_seq_all, out='y', phase=phase)
            rot_seq_all = self.update_outputs_batch(length, nodes_all[b], nodes_sem_labels,
                                                    rot_seq_all, out='rot', phase=phase)
        y_outs = out_seq_to_batch(lengths, y_seq_all)
        rot_outs = out_seq_to_batch(lengths, rot_seq_all)

        outputs = wrap_up_outputs(self.xyz, lengths, inputs, y_outs, rot_outs, phase=phase)
        return outputs

    def update_outputs_batch(self, length, nodes, nodes_sem_labels, seq_all, out=None, phase=None):
        seq = [0 for _ in range(length)]  # 11
        for i in range(length):
            if opt.sem_select == 'gt' and phase == 'train':
                sem_i = nodes_sem_labels[i][0].item()
            else:
                cls = nodes_sem_labels[i]   # (1, 7)
                _, sem_i = torch.max(cls[0, :], 0)
                sem_i = sem_i.item()
            if out == 'y':
                output = self.y_linear_final[sem_i](nodes[i])
            elif out == 'rot':
                output = self.rot_linear_final[sem_i](nodes[i])
            else:
                raise NotImplementedError
            seq[i] = output  # 1, 6
        seq_all.append(seq)
        return seq_all


class GraphStack(nn.Module):
    def __init__(self, node, embed_type, g1_in_size, xyz):
        super(GraphStack, self).__init__()
        if opt.graph == 'design':
            self.embed = NodeEmbed(node, embed_type, g1_in_size, xyz)
            self.layers = nn.ModuleList([GraphLayer() for _ in range(opt.mp)])
        elif opt.graph == 'base':
            # self.embed = NodeEmbedBase(g1_in_size)
            self.embed = NodeEmbed(node, embed_type, g1_in_size, xyz)
            self.layers = nn.ModuleList([GraphLayerBase() for _ in range(opt.mp)])
        else:
            raise NotImplementedError

    def forward(self, inputs):
        nodes = self.embed(inputs)
        for i in range(opt.mp):
            nodes = self.layers[i](nodes, inputs)
        return nodes


class GraphLayer(nn.Module):
    def __init__(self):
        super(GraphLayer, self).__init__()
        if opt.mes_type == '0':
            self.g3 = nn.Sequential(
                nn.Linear(opt.f_dim, opt.f_dim),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
            self.g4 = nn.Sequential(
                nn.Linear(opt.f_dim, opt.f_dim),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        elif opt.mes_type == '1':       ## Nan
            self.g3 = nn.Sequential(
                nn.Linear(opt.f_dim * 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        elif opt.mes_type == '2':
            self.g3 = nn.Sequential(
                nn.Linear(opt.f_dim * 2, opt.f_dim),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        elif opt.mes_type == '3':
            self.g3 = nn.Sequential(
                nn.Linear(opt.f_dim, opt.f_dim),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
            self.g4 = nn.Sequential(
                nn.Linear(opt.f_dim, opt.f_dim),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
            self.g6 = nn.Sequential(
                nn.Linear(opt.f_dim, opt.f_dim),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        elif opt.mes_type == '4':
            self.g3 = nn.Sequential(
                nn.Linear(opt.f_dim * 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
            self.g6 = nn.Sequential(
                nn.Linear(opt.f_dim, opt.f_dim),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        elif opt.mes_type == '5':
            self.g3 = nn.Sequential(
                nn.Linear(opt.f_dim * 2, opt.f_dim),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
            self.g6 = nn.Sequential(
                nn.Linear(opt.f_dim, opt.f_dim),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        elif opt.mes_type == '6':
            self.g3 = nn.Sequential(
                nn.Linear(opt.f_dim * 2, opt.f_dim),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
            self.g7 = nn.Sequential(
                nn.Linear(opt.f_dim * 2, 1),
                nn.Sigmoid()
            )
        self.g5 = nn.Sequential(
            nn.Linear(opt.f_dim, opt.f_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def compute_message(self, n_i, n_j):
        if opt.mes_type == '0':
            w1 = self.g3(n_i)
            w2 = self.g4(n_j)
            weight = torch.dot(w1[0], w2[0])
            message = weight * n_j
        elif opt.mes_type == '1':
            w = torch.cat((n_i, n_j), dim=1)
            weight = self.g3(w)
            message = weight * n_j
        elif opt.mes_type == '2':
            w = torch.cat((n_i, n_j), dim=1)
            message = self.g3(w)
        elif opt.mes_type == '3':
            w1 = self.g3(n_i)
            w2 = self.g4(n_j)
            weight = torch.dot(w1[0], w2[0])
            message = weight * self.g6(n_j)
        elif opt.mes_type == '4':
            w = torch.cat((n_i, n_j), dim=1)
            weight = self.g3(w)
            message = weight * self.g6(n_j)
        elif opt.mes_type == '5':
            w = torch.cat((n_i, n_j), dim=1)
            message = self.g6(self.g3(w))
        elif opt.mes_type == '6':
            w = torch.cat((n_i, n_j), dim=1)
            a = self.g7(w)
            message = a * self.g3(w)
        else:
            message = 0.
        return message

    def forward(self, nodes_in, features=None):
        nodes_out = []
        num = len(nodes_in)
        if num <= 1:
            return nodes_in
        for i in range(num):
            message = 0
            for j in range(num):
                if j != i:
                    message += self.compute_message(nodes_in[i], nodes_in[j]) / (num - 1)
            node = self.g5(message) + nodes_in[i]
            nodes_out.append(node)
        return nodes_out


class NodeEmbed(nn.Module):
    def __init__(self, node, embed_type, g1_in_size, xyz):
        super(NodeEmbed, self).__init__()
        self.xyz = xyz
        self.node = node
        size_in = {'img': get_img_feature_size(self.node),
                    'h': opt.hid_size * opt.hid_layer,
                    'n': opt.f_dim,
                    'st': 2 * self.xyz,
                    'r': 1 * self.xyz,
                    'cls': opt.n_sem + int(opt.len_adjust),
                    'cot': opt.n_sem + int(opt.len_adjust)}
        size_out = {}
        if embed_type == '0':
            size_out['h'] = opt.f_dim
        if embed_type == '1':
            size_out['h'] = int(opt.f_dim * 0.9)
            size_out['cot'] = opt.f_dim - size_out['h']
        if embed_type == '2':
            size_out['h'] = int(opt.f_dim * 0.9)
            size_out['cls'] = opt.f_dim - size_out['h']
        if embed_type == '3':
            size_out['h'] = int(opt.f_dim * 0.8)
            size_out['cot'] = opt.f_dim - size_out['h']
        if embed_type == '4':
            size_out['h'] = int(opt.f_dim * 0.8)
            size_out['cls'] = opt.f_dim - size_out['h']
        if embed_type == '5':
            size_out['h'] = int(opt.f_dim * 0.7)
            size_out['cot'] = opt.f_dim - size_out['h']
        if embed_type == '6':
            size_out['h'] = int(opt.f_dim * 0.7)
            size_out['cls'] = opt.f_dim - size_out['h']
        if embed_type == '7':
            size_out['h'] = int(opt.f_dim * 0.95)
            size_out['cot'] = opt.f_dim - size_out['h']
        if embed_type == '8':
            size_out['h'] = int(opt.f_dim * 0.95)
            size_out['cls'] = opt.f_dim - size_out['h']

        if embed_type == '10':
            size_out['h'] = int(opt.f_dim * 0.9)
            size_out['cot'] = int(opt.f_dim * 0.05)
            size_out['st'] = opt.f_dim - size_out['h'] - size_out['cot']
        if embed_type == '11':
            size_out['h'] = int(opt.f_dim * 0.85)
            size_out['cot'] = int(opt.f_dim * 0.1)
            size_out['st'] = opt.f_dim - size_out['h'] - size_out['cot']
        if embed_type == '12':
            size_out['h'] = int(opt.f_dim * 0.8)
            size_out['cot'] = int(opt.f_dim * 0.1)
            size_out['st'] = opt.f_dim - size_out['h'] - size_out['cot']

        if embed_type == '20':
            size_out['h'] = int(opt.f_dim * 0.9)
            size_out['cls'] = int(opt.f_dim * 0.05)
            size_out['st'] = opt.f_dim - size_out['h'] - size_out['cls']
        if embed_type == '21':
            size_out['h'] = int(opt.f_dim * 0.85)
            size_out['cls'] = int(opt.f_dim * 0.1)
            size_out['st'] = opt.f_dim - size_out['h'] - size_out['cls']
        if embed_type == '22':
            size_out['h'] = int(opt.f_dim * 0.8)
            size_out['cls'] = int(opt.f_dim * 0.1)
            size_out['st'] = opt.f_dim - size_out['h'] - size_out['cls']

        if embed_type == '30':
            size_out['h'] = int(opt.f_dim * 0.9)
            size_out['cls'] = int(opt.f_dim * 0.05)
            size_out['st'] = int(opt.f_dim * 0.035)
            size_out['r'] = opt.f_dim - size_out['h'] - size_out['cls'] - size_out['st']
        if embed_type == '31':
            size_out['h'] = int(opt.f_dim * 0.85)
            size_out['cls'] = int(opt.f_dim * 0.1)
            size_out['st'] = int(opt.f_dim * 0.035)
            size_out['r'] = opt.f_dim - size_out['h'] - size_out['cls'] - size_out['st']
        if embed_type == '32':
            size_out['h'] = int(opt.f_dim * 0.8)
            size_out['cls'] = int(opt.f_dim * 0.1)
            size_out['st'] = int(opt.f_dim * 0.07)
            size_out['r'] = opt.f_dim - size_out['h'] - size_out['cls'] - size_out['st']
        self.keys = self.node.split('_')
        if 'n' in self.keys:
            size_out['n'] = size_out['h']
        self.embed_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(size_in[key], size_out[key]),
            nn.ReLU()
            # nn.LeakyReLU(negative_slope=0.1, inplace=True)
        ) for key in self.keys])
        size_sum = 0
        for key in self.keys:
            size_sum += size_out[key]
        if opt.embed_fuse > 0:
            self.fuse = nn.Sequential(
                nn.Linear(size_sum, opt.f_dim),
                nn.ReLU()
            )

    def forward(self, inputs, features=None):
        nodes = []
        num = len(inputs)
        for i in range(num):
            node, cls_one_hot = inputs[i]   # (1, 9), (1, 6)
            node_i = []
            for j in range(len(self.keys)):
                key = self.keys[j]
                node_j = self.embed_layers[j](node[key])
                node_i.append(node_j)
            node = torch.cat(node_i, dim=1)
            if opt.embed_fuse > 0:
                node = self.fuse(node)
            nodes.append(node)
        return nodes


class GraphLayerBase(nn.Module):
    def __init__(self):
        super(GraphLayerBase, self).__init__()
        self.dis_criterion = torch.nn.L1Loss()
        if opt.mes_type == '0':
            self.g2 = nn.Linear(opt.f_dim, opt.f_dim)
            self.g3 = nn.Linear(opt.f_dim, opt.f_dim)
            self.g4 = nn.Linear(opt.f_dim, opt.f_dim)
            self.g5 = nn.Linear(opt.f_dim, opt.f_dim)
        elif opt.mes_type == '1':
            self.g2 = nn.Linear(opt.f_dim, opt.f_dim)
            self.g3 = nn.Linear(opt.f_dim * 2, opt.f_dim)
            self.relu = nn.ReLU()
            self.g5 = nn.Linear(opt.f_dim, opt.f_dim)
        elif opt.mes_type == '2':   ##best, converges slower
            self.g2 = nn.Linear(opt.f_dim, opt.f_dim)
            self.g3 = nn.Linear(opt.f_dim, opt.f_dim)
            self.g5 = nn.Linear(opt.f_dim, opt.f_dim)
        elif opt.mes_type == '3':
            self.g3 = nn.Linear(opt.f_dim, opt.f_dim)
            self.g5 = nn.Linear(opt.f_dim, opt.f_dim)
        elif opt.mes_type == '4':
            self.g3 = nn.Linear(opt.f_dim, opt.f_dim)
            self.g5 = nn.ReLU()
        elif opt.mes_type == '5':
            self.g3 = nn.Linear(opt.f_dim, opt.f_dim)
            self.g5 = nn.Sequential(
                nn.Linear(opt.f_dim, opt.f_dim),
                nn.ReLU()
            )
        elif opt.mes_type == '6':   ## from 2
            self.g2 = nn.Linear(opt.f_dim, opt.f_dim)
            self.g3 = nn.Linear(opt.f_dim, opt.f_dim)
            self.g5 = nn.Linear(opt.f_dim, opt.f_dim)
            self.softmax = nn.Softmax(dim=1)

    def compute_weight(self, n_i, n_j):
        if opt.mes_type == '0':
            w1 = self.g3(n_i)
            w2 = self.g4(n_j)
            weight = torch.dot(w1[0], w2[0])
        elif opt.mes_type == '1':
            w = torch.cat((n_i, n_j), dim=1)
            weight = self.relu(self.g3(w))
        elif opt.mes_type in ['2', '3', '4', '5', '6']:
            w1 = self.g3(n_i)
            w2 = self.g3(n_j)
            # pdb.set_trace()
            weight = torch.dot(w1[0], w2[0])
        else:
            weight = 1.
        return weight

    def forward(self, nodes_in, inputs, features=None):
        nodes_out = []
        num = len(nodes_in)
        if num <= 1:
            return nodes_in
        for i in range(num):
            info = collections.OrderedDict()
            dis_i = []
            for j in range(num):
                if j != i:
                    weight = self.compute_weight(nodes_in[i], nodes_in[j])
                    info[j] = [weight]
                    if 'st' in opt.partial_graph:
                        prim_center_i = inputs[i][0]['st'][0, :3] + inputs[i][0]['st'][0, 3:]
                        prim_center_j = inputs[j][0]['st'][0, :3] + inputs[j][0]['st'][0, 3:]
                        distance = (prim_center_i - prim_center_j).abs().sum()
                        info[j] = [weight, distance]
                        dis_i.append(distance.item())
                    if 'cot' in opt.partial_graph:
                        _, prim_sem_i = torch.max(inputs[i][0]['cot'], dim=1)
                        _, prim_sem_j = torch.max(inputs[j][0]['cot'], dim=1)
                        # print(prim_sem_i,prim_sem_j)
                        if prim_sem_i.item() != 1 and prim_sem_j.item() != 1 and opt.obj_class == 'chair':
                            info.pop(j)
            if 'st' in opt.partial_graph:
                select = 3
                idxs = torch.argsort(torch.Tensor(dis_i))
                idxs = idxs[:select].tolist()
                info_keys = [k for k, _ in info.items()]
                for j in info_keys:
                    if j not in idxs:
                        info.pop(j)
            if opt.mes_type == '6':
                pdb.set_trace()
                weight_i = [x[0] for _, x in info.items()]
                weight_i = torch.cat(weight_i, dim=1)
                weight_i = self.softmax(weight_i)
                for c in range(len(info.keys())):
                    key = info.keys()[c]
                    info[key] = [weight_i[c]]

            message = 0
            info_keys = [k for k, _ in info.items()]
            if len(info_keys) > 0:
                for j in range(num):
                    # if j != i:
                    if j in info_keys:
                        weight_ij = info[j][0]
                        if opt.mes_type in ['0', '1', '2']:
                            message_j = weight_ij * self.g2(nodes_in[j])
                        elif opt.mes_type in ['3', '4', '5']:
                            message_j = weight_ij * self.g3(nodes_in[j])
                        else:
                            raise NotImplementedError
                        message += message_j / len(info.keys())
                node = self.g5(message) + nodes_in[i]
            else:
                node = nodes_in[i]
            nodes_out.append(node)
        return nodes_out


class NodeEmbedBase(nn.Module):
    def __init__(self, g1_in_size):
        super(NodeEmbedBase, self).__init__()
        self.g1 = nn.Sequential(
            nn.Linear(g1_in_size + opt.n_sem + int(opt.len_adjust), opt.f_dim),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, inputs, features=None):
        nodes = []
        num = len(inputs)
        for i in range(num):
            node, cls_one_hot = inputs[i]   # (1, opt.hid_size * opt.hid_layer), (1, 6/7)
            node = torch.cat((node, cls_one_hot), dim=1)
            node = self.g1(node)    # # (1, opt.f_dim)
            nodes.append(node)
        return nodes


class GATPrim(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GATPrim, self).__init__()
        global opt
        opt = get_opt()
        self.nclass = nclass
        self.gat = GAT(nfeat, nhid, nclass, dropout, alpha, nheads)
        in_size = 3*self.xyz
        if opt.loss_box2d is not None:
            in_size += 4
        if opt.loss_c is not None:
            in_size += opt.n_sem
        self.up_linear = nn.Linear(in_size, opt.con_size)
        if self.nclass > 10:
            if opt.sem_reg:
                self.y_linear = nn.ModuleList([nn.Linear(nclass, 2 * self.xyz) for _ in range(opt.n_sem)])
                self.r_linear = nn.ModuleList([nn.Linear(nclass, 1 * self.xyz) for _ in range(opt.n_sem)])
                # self.cls_linear = nn.Linear(nclass, opt.n_sem)
                # self.box2d_linear = nn.ModuleList([nn.Linear(nclass, 4) for _ in range(opt.n_sem)])
            else:
                self.y_linear = nn.Linear(nclass, 2*self.xyz)
                self.r_linear = nn.Linear(nclass, 1*self.xyz)
                # self.cls_linear = nn.Linear(nclass, opt.n_sem)
            self.box2d_linear = nn.Linear(nclass, 4)

    def forward(self, features, outputs, adj):   #x, adj):
        # import pdb;pdb.set_trace()
        # print('-' * 30, features[0].size(), features[0][0])
        # print('*' * 30, outputs)
        assert len(features) == len(outputs)    # 33-1/29/..., 16, 2/1/4
        graphs = []
        for i in range(len(features)):
            # import pdb;pdb.set_trace()
            graphs_i = torch.cat((outputs[i][0], outputs[i][1]), dim=1)  # 16, 3
            if opt.loss_c is not None:
                _, cls_id = torch.max(outputs[i][2],dim=1)
                cls_one_hot = torch.zeros(outputs[i][2].size())
                for ii in range(cls_one_hot.size(0)):
                    cls_one_hot[ii, cls_id[ii]] = 1
                cls_one_hot = cls_one_hot.cuda()
                graphs_i = torch.cat((graphs_i, cls_one_hot), dim=1)  # 16, +opt.n_sem
            if opt.loss_box2d is not None:
                graphs_i = torch.cat((graphs_i, outputs[i][3]), dim=1)  # 16, +4
            graphs_i = self.up_linear(graphs_i) # 16, 256
            # import pdb
            # pdb.set_trace()
            if 'pred' in opt.feature_scale:
                graphs_i = graphs_i / graphs_i.norm(dim=1).unsqueeze(dim=1)
            graphs_i = torch.cat((features[i], graphs_i), dim=1)   # 16, 256+32+256
            graphs.append(graphs_i.unsqueeze(dim=1))    # 16, 1, 544
        graphs = torch.cat(graphs, dim=1)   # 16, 33, 544
        # pdb.set_trace()
        out_all = []
        for i in range(graphs.size(0)):
            out_i = self.gat(graphs[i], adj[i])
            out_all.append(out_i.unsqueeze(dim=0))
            # print(i, graphs[i].size(), out_i.size())
        out_all = torch.cat(out_all, dim=0) # 16, 33, 7
        # pdb.set_trace()
        outs = []
        # print(len(features), out_all.size())
        assert len(features) == out_all.size(1)
        for i in range(out_all.size(1)):
            if self.nclass > 10:
                if opt.sem_reg:
                    y = self.forward_by_ins(out_all[:, i, :], outputs[i][2], out='y')
                    r = self.forward_by_ins(out_all[:, i, :], outputs[i][2], out='rot')
                else:
                    y = self.y_linear(out_all[:, i, :])
                    r = self.r_linear(out_all[:, i, :])
                # cls = self.cls_linear(out_all[:, i, :])
                box2d = self.box2d_linear(out_all[:, i, :])
            else:
                y = out_all[:, i, :2]
                r = out_all[:, i, 2:3]
                # cls = out_all[:, i, :]
                box2d = out_all[:, i, 3:]
            outs.append((y, r, outputs[i][2], box2d))
        # print('=' * 30, outs)
        return outs

    def forward_by_ins(self, h, cls, out=None):
        if h.size(0) <= self.n_sem:
            return self.forward_by_loop(h, cls, out)
        else:
            return self.forward_by_mask(h, cls, out)

    def forward_by_loop(self, h, cls, out=None):
        y = []
        for ins_i in range(cls.size(0)):
            _, cls_id = torch.max(cls[ins_i, :], 0)
            cls_id = cls_id.item()
            if out == 'y':
                y.append(self.y_linear[cls_id](h[ins_i:ins_i+1]))
            elif out == 'rot':
                y.append(self.r_linear[cls_id](h[ins_i:ins_i+1]))
            else:
                raise NotImplementedError
        y = torch.cat(y, dim=0)
        return y

    def forward_by_mask(self, h, cls, out=None):
        pass


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        if opt.n_att > 0:
            self.atts = [[GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True)
                         for _ in range(nheads)] for _ in range(opt.n_att)]
            for ii in range(opt.n_att):
                for i, att in enumerate(self.atts[ii]):
                    self.add_module('att_{}_{}'.format(ii, i), att)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

        self.out_atts = [GraphAttentionLayer(nclass, nclass, dropout=dropout, alpha=alpha, concat=False)
                         for _ in range(opt.n_graph - 2)]
        for i, att in enumerate(self.out_atts):
            self.add_module('out_att_{}'.format(i), att)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        if opt.n_att > 0:
            for ii in range(opt.n_att):
                x = torch.cat([att(x, adj) for att in self.atts[ii]], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        if len(self.out_atts) > 0:
            for i in range(len(self.out_atts)):
                x = F.elu(self.out_atts[i](x, adj))
        # return F.log_softmax(x, dim=1)
        return x


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        stdv = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.W.data, -stdv, stdv)
        # self.ww = nn.Linear(in_features, out_features)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)
        stdv = 1.0 / math.sqrt(2 * out_features)
        nn.init.uniform_(self.a.data, -stdv, stdv)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    # def reset_parameters(self):
    #     nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
        #     bound = 1 / math.sqrt(fan_in)
        #     nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        # h = self.ww(input)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # import pdb;pdb.set_trace()
        attention = F.dropout(attention, self.dropout, training=self.training)
        if np.random.random() < 0.01:
            print(attention.size())
            print(attention[np.random.randint(0, attention.size(0))])
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
