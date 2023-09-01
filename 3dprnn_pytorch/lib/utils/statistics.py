import os
import time
import datetime
import copy
import visdom
import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from lib.opts import *


class StatisticsPrnn(object):
    def __init__(self):
        global opt
        opt = get_opt()
        self.since = time.time()

        self.lowest_val_loss = 1e8
        self.lowest_val_loss_x = 1e8
        self.lowest_val_loss_r = 1e8
        self.lowest_val_loss_r_theta = 1e8
        self.lowest_val_loss_r_axis = 1e8
        self.lowest_val_loss_c = 1e8
        self.lowest_val_loss_b = 1e8
        self.lowest_val_loss_e = 1e8
        self.lowest_val_loss_s = 1e8
        self.lowest_val_loss_box2d = 1e8
        self.running_loss = 0.
        self.running_loss_x = 0.
        self.running_loss_r = 0.
        self.running_loss_r_theta = 0.
        self.running_loss_r_axis = 0.
        self.running_loss_c = 0.
        self.running_loss_b = 0.
        self.running_loss_e = 0.
        self.running_loss_s = 0.
        self.running_loss_box2d = 0.
        self.epoch_loss = 0.
        self.epoch_loss_x = 0.
        self.epoch_loss_r = 0.
        self.epoch_loss_r_theta = 0.
        self.epoch_loss_r_axis = 0.
        self.epoch_loss_c = 0.
        self.epoch_loss_b = 0.
        self.epoch_loss_e = 0.
        self.epoch_loss_s = 0.
        self.epoch_loss_box2d = 0.

        # if opt.save_nn:
        #     self.d = None
        #     self.init = None

        # self.best_iou = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        # self.best_acc = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        # self.inexistent = {x: 0 for x in list(range(opt.n_sem)) + ['mean']}     # inexistent part batch count
        # self.running_iou = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        # self.running_acc = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        # self.epoch_iou = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        # self.epoch_acc = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        #
        # self.threshes = [0.3, 0.4, 0.5]#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # self.gt_prim_sum = {x: 0 for x in list(range(opt.n_sem)) + ['mean']}
        # self.prob_iou_exist_id = {x: [] for x in list(range(opt.n_sem)) + ['mean']}
        # self.precision = {x: {y: [] for y in self.threshes} for x in list(range(opt.n_sem)) + ['mean']}
        # self.recall = {x: {y: [] for y in self.threshes} for x in list(range(opt.n_sem)) + ['mean']}
        # self.avg_precision = {x: {y: 0 for y in self.threshes} for x in list(range(opt.n_sem)) + ['mean']}

        # if opt.demo is not None:
        #     self.result_parts = None
        #     self.result_existence = None
        if opt.visdom:
            if opt.machine == 'mac_h':
                self.vis = visdom.Visdom(env=opt.env, server='http://localhost', port=8097)
            elif opt.machine == 'ai':
                if opt.vis_node == 'admin':
                    self.vis = visdom.Visdom(env=opt.env, server='http://10.10.10.100', port=31723)
                else:
                    self.vis = visdom.Visdom(env=opt.env, server='http://10.10.10.1{}'.format(opt.vis_node), port=31721)
                # self.vis = visdom.Visdom(env=opt.env, server='http://10.10.10.121', port=31723) # node21
                # node21 need to run: unset https_proxy; unset http_proxy
            else:
                raise NotImplementedError
            if opt.demo is None:
                if not opt.test_curve and not opt.nn_curve and not opt.test_through:
                    self.vis.close(win=None, env=opt.env)

    def reset_epoch(self, phase):
        self.running_loss = 0.
        self.running_loss_x = 0.
        self.running_loss_r = 0.
        self.running_loss_r_theta = 0.
        self.running_loss_r_axis = 0.
        self.running_loss_c = 0.
        self.running_loss_b = 0.
        self.running_loss_e = 0.
        self.running_loss_s = 0.
        self.running_loss_box2d = 0.
        self.epoch_loss = 0.
        self.epoch_loss_x = 0.
        self.epoch_loss_r = 0.
        self.epoch_loss_r_theta = 0.
        self.epoch_loss_r_axis = 0.
        self.epoch_loss_c = 0.
        self.epoch_loss_b = 0.
        self.epoch_loss_e = 0.
        self.epoch_loss_s = 0.
        self.epoch_loss_box2d = 0.

        # self.inexistent = {x: 0 for x in list(range(opt.n_sem)) + ['mean']}     # inexistent part batch count
        # self.running_iou = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        # self.running_acc = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        # self.epoch_iou = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        # self.epoch_acc = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        #
        # if phase == 'val':
        #     self.gt_prim_sum = {x: 0 for x in list(range(opt.n_sem)) + ['mean']}
        #     self.prob_iou_exist_id = {x: [] for x in list(range(opt.n_sem)) + ['mean']}
        #     self.precision = {x: {y: [] for y in self.threshes} for x in list(range(opt.n_sem)) + ['mean']}
        #     self.recall = {x: {y: [] for y in self.threshes} for x in list(range(opt.n_sem)) + ['mean']}
        #     self.avg_precision = {x: {y: 0 for y in self.threshes} for x in list(range(opt.n_sem)) + ['mean']}

    def accumulate(self, loss, loss_x, loss_r, loss_c, loss_b, loss_e, loss_s, loss_box2d):
        self.running_loss += loss.item()
        self.running_loss_x += loss_x.item()
        if opt.out_r == 'class':
            loss_r_theta, loss_r_axis = loss_r
            self.running_loss_r_theta += loss_r_theta.item()
            self.running_loss_r_axis += loss_r_axis.item()
            self.running_loss_r = self.running_loss_r_theta + self.running_loss_r_axis
        else:
            self.running_loss_r += loss_r.item()
        self.running_loss_c += loss_c.item()
        self.running_loss_b += loss_b.item()
        self.running_loss_e += loss_e.item()
        self.running_loss_s += loss_s.item()
        self.running_loss_box2d += loss_box2d.item()
        # for key in self.running_iou.keys():
        #     if iou[key].item() > 0:
        #         self.running_iou[key] += iou[key].item()
        #     else:
        #         self.inexistent[key] += 1###
        #     self.running_acc[key] += acc[key].item()
        #
        #     if phase == 'val' or opt.demo is not None:
        #         if opt.exist and key != 'mean':
        #             _, out_exist = outputs
        #             self.prob_iou_exist_id[key].append((out_exist[0, key].item(), iou[key].item(),
        #                                           existence[0, key].item(), i_batch))##

    def compute_epoch(self, i_batch, phase):
        self.epoch_loss = self.running_loss / (i_batch + 1)
        self.epoch_loss_x = self.running_loss_x / (i_batch + 1)
        self.epoch_loss_r = self.running_loss_r / (i_batch + 1)
        self.epoch_loss_r_theta = self.running_loss_r_theta / (i_batch + 1)
        self.epoch_loss_r_axis = self.running_loss_r_axis / (i_batch + 1)
        self.epoch_loss_c = self.running_loss_c / (i_batch + 1)
        self.epoch_loss_b = self.running_loss_b / (i_batch + 1)
        self.epoch_loss_e = self.running_loss_e / (i_batch + 1)
        self.epoch_loss_s = self.running_loss_s / (i_batch + 1)
        self.epoch_loss_box2d = self.running_loss_box2d / (i_batch + 1)
        # for key in self.epoch_iou.keys():
        #     self.epoch_iou[key] = self.running_iou[key] / (i_batch + 1 - self.inexistent[key])
        #     self.epoch_acc[key] = self.running_acc[key] / (i_batch + 1)
        # if phase == 'val' or opt.demo is not None:
        #     self.evaluate_precision_recall_ap()

    def update_best(self):
        self.lowest_val_loss = self.epoch_loss
        self.lowest_val_loss_x = self.epoch_loss_x
        self.lowest_val_loss_r = self.epoch_loss_r
        self.lowest_val_loss_r_theta = self.epoch_loss_r_theta
        self.lowest_val_loss_r_axis = self.epoch_loss_r_axis
        self.lowest_val_loss_c = self.epoch_loss_c
        self.lowest_val_loss_b = self.epoch_loss_b
        self.lowest_val_loss_e = self.epoch_loss_e
        self.lowest_val_loss_s = self.epoch_loss_s
        self.lowest_val_loss_box2d = self.epoch_loss_box2d
        # self.best_iou = copy.deepcopy(self.epoch_iou)
        # self.best_acc = copy.deepcopy(self.epoch_acc)

    def is_best(self, i_batch, phase):
        self.compute_epoch(i_batch, phase)
        if (phase == 'val' or phase == 'train_val') and self.epoch_loss < self.lowest_val_loss:
            self.update_best()
            return True
        else:
            return False

    def print_batch(self, epoch, i_batch):
        print('Epoch: {:4.0f} i_batch: {:4.0f} Loss_x: {:.6f} Loss_r_theta: {:.6f} Loss_r_axis: {:.6f} Loss_r: {:.6f} '
              'Loss_c: {:.6f} Loss_b: {:.6f} Loss_e: {:.6f} Loss_s: {:.6f} Loss_box2d: {:.6f} Loss: {:.6f}'.format(
            epoch, i_batch, self.running_loss_x / (i_batch + 1),
            self.running_loss_r_theta / (i_batch + 1), self.running_loss_r_axis / (i_batch + 1),
            self.running_loss_r / (i_batch + 1), self.running_loss_c / (i_batch + 1),
            self.running_loss_b / (i_batch + 1), self.running_loss_e / (i_batch + 1),
            self.running_loss_s / (i_batch + 1), self.running_loss_box2d / (i_batch + 1),
            self.running_loss / (i_batch + 1)))

    def print_epoch_begin(self, epoch):
        print('Epoch {}/{}'.format(epoch, opt.n_epochs) + '-' * 30)
        print('Time: %s' % datetime.datetime.now())

    def print_epoch_end(self, epoch, phase):
        print('{} Epoch: {:4.0f} Loss: {:.8f}'.format(phase, epoch, self.epoch_loss))
        print('lowest_val_r_theta: {:.8f} lowest_val_r_axis: {:.8f} '
              'lowest_val_x: {:.8f} lowest_val_r: {:.8f} '
              'lowest_val_c: {:.8f} lowest_val_b: {:.8f} '
              'lowest_val_e: {:.8f} lowest_val_s: {:.8f} '
              'lowest_val_box2d: {:.8f} lowest_val_loss: {:.8f}'.format(
            self.lowest_val_loss_r_theta, self.lowest_val_loss_r_axis,
            self.lowest_val_loss_x, self.lowest_val_loss_r, self.lowest_val_loss_c,
            self.lowest_val_loss_b, self.lowest_val_loss_e, self.lowest_val_loss_s,
            self.lowest_val_loss_box2d, self.lowest_val_loss))
        time_elapsed = time.time() - self.since
        print('Time till now {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 60 // 60, time_elapsed // 60 % 60, time_elapsed % 60))

    def print_final(self):
        time_elapsed = time.time() - self.since
        print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 60 // 60, time_elapsed // 60 % 60, time_elapsed % 60))

    def plot(self, epoch, phase, is_best_tag=False):
        # if epoch == 0 and (phase == 'train' or phase == 'train_val'):
            # self.plot_torch_loss()
        if opt.demo is None:
            self.plot_loss(epoch, phase)
        #     self.plot_eval(epoch, phase, 'mean_iou')
        #     self.plot_eval(epoch, phase, 'mean_acc')
        #     self.plot_iou_acc(epoch, phase, 'iou')
        #     self.plot_iou_acc(epoch, phase, 'acc')
        # if (phase == 'val' and is_best_tag) or opt.demo is not None:
        #     self.plot_precision_recall_ap()
        self.vis.save(envs=[opt.env])

    def plot_loss(self, epoch, phase):
        self.plot_loss_one('Loss_all', epoch, phase)
        self.plot_loss_one('Loss_x', epoch, phase)
        if opt.out_r == 'class':
            self.plot_loss_one('Loss_r_theta', epoch, phase)
            self.plot_loss_one('Loss_r_axis', epoch, phase)
        self.plot_loss_one('Loss_r', epoch, phase)
        self.plot_loss_one('Loss_c', epoch, phase)
        if opt.bbox_loss != 'None':
            self.plot_loss_one('Loss_b', epoch, phase)
        if opt.loss_e is not None:
            self.plot_loss_one('Loss_e', epoch, phase)
        if opt.sigma_reg > 0.:
            self.plot_loss_one('Loss_s', epoch, phase)
        if opt.loss_box2d is not None:
            self.plot_loss_one('Loss_box2d', epoch, phase)

    def plot_loss_one(self, win_name, epoch, phase, value=None):
        if value is None:
            if win_name == 'Loss_all':
                value = self.epoch_loss
            elif win_name == 'Loss_x':
                value = self.epoch_loss_x
            elif win_name == 'Loss_r':
                value = self.epoch_loss_r
            elif win_name == 'Loss_r_theta':
                value = self.epoch_loss_r_theta
            elif win_name == 'Loss_r_axis':
                value = self.epoch_loss_r_axis
            elif win_name == 'Loss_c':
                value = self.epoch_loss_c
            elif win_name == 'Loss_b':
                value = self.epoch_loss_b
            elif win_name == 'Loss_e':
                value = self.epoch_loss_e
            elif win_name == 'Loss_s':
                value = self.epoch_loss_s
            elif win_name == 'Loss_box2d':
                value = self.epoch_loss_box2d
            else:
                raise NotImplementedError
        self.vis.line(Y=torch.Tensor([value]), X=torch.Tensor([epoch]),
                      win=win_name, env=opt.env,
                      opts=dict(title=win_name, xlabel='epoch', showlegend=True),
                      update='append',
                      name=phase)

    def plot_torch_loss(self):
        import torchfile
        import os
        if opt.machine == 'mac_h':
            path = '/Users/heqian/Research/1112/projects/3dprnn/intact/torch_test/'# + opt.env
            # path = '/Users/heqian/Research/projects/3dprnn/3d1/network/setting/old/losses'
        elif opt.machine == 'ai':
            path = '/root/projects/3dprnn/intact/torch9'
        else:
            raise NotImplementedError
        # if not opt.depth_con:
        if opt.encoder is None:
            file_name = 'loss_to_save.t7'
        else:
            file_name = 'loss_to_save_depth.t7'
        loss = torchfile.load(os.path.join(path, file_name))
        for i in range(len(loss[b'train'].keys())):
            self.plot_loss_one('Loss_all_torch', i, 'train', loss[b'train'][i])
            # self.plot_loss_one('Loss_x_torch', i, 'train', loss[b'train_x'][i])
            # self.plot_loss_one('Loss_r_torch', i, 'train', loss[b'train_r'][i])
            self.plot_loss_one('Loss_all_torch', i, 'val', loss[b'val'][i])
            # self.plot_loss_one('Loss_x_torch', i, 'val', loss[b'val_x'][i])
            # self.plot_loss_one('Loss_r_torch', i, 'val', loss[b'val_r'][i])

    def plot_iou_one(self, epoch, value, phase, win_name):
        # win_name = 'IoU'
        # print(epoch, value, phase, win_name)
        self.vis.line(Y=torch.Tensor([value]), X=torch.Tensor([epoch]),
                      win=win_name, env=opt.env,
                      opts=dict(title=win_name, xlabel='epoch', showlegend=True),
                      update='append',
                      name=phase[:11])
        self.vis.save(envs=[opt.env])


class StatisticsBbox(object):
    def __init__(self):
        global opt
        opt = get_opt()
        self.since = time.time()

        self.lowest_val_loss = 1e8
        self.lowest_val_loss_regress = 1e8
        self.lowest_val_loss_exist = 1e8
        self.running_loss = 0.
        self.running_loss_regress = 0.
        self.running_loss_exist = 0.
        self.epoch_loss = 0.
        self.epoch_loss_regress = 0.
        self.epoch_loss_exist = 0.

        self.best_iou = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        self.best_acc = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        self.inexistent = {x: 0 for x in list(range(opt.n_sem)) + ['mean']}     # inexistent part batch count
        self.running_iou = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        self.running_acc = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        self.epoch_iou = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        self.epoch_acc = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}

        self.threshes = [0.3, 0.4, 0.5]#[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.gt_prim_sum = {x: 0 for x in list(range(opt.n_sem)) + ['mean']}
        self.prob_iou_exist_id = {x: [] for x in list(range(opt.n_sem)) + ['mean']}
        self.precision = {x: {y: [] for y in self.threshes} for x in list(range(opt.n_sem)) + ['mean']}
        self.recall = {x: {y: [] for y in self.threshes} for x in list(range(opt.n_sem)) + ['mean']}
        self.avg_precision = {x: {y: 0 for y in self.threshes} for x in list(range(opt.n_sem)) + ['mean']}

        if opt.demo is not None:
            self.result_parts = None
            self.result_existence = None
        if opt.visdom:
            self.vis = visdom.Visdom(env=opt.env, server='http://10.10.10.100', port=31723)
            if opt.demo is None:
                self.vis.close(win=None, env=opt.env)

    def reset_epoch(self, phase):
        self.running_loss = 0.
        self.running_loss_regress = 0.
        self.running_loss_exist = 0.
        self.epoch_loss = 0.
        self.epoch_loss_regress = 0.
        self.epoch_loss_exist = 0.

        self.inexistent = {x: 0 for x in list(range(opt.n_sem)) + ['mean']}     # inexistent part batch count
        self.running_iou = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        self.running_acc = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        self.epoch_iou = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}
        self.epoch_acc = {x: 0. for x in list(range(opt.n_sem)) + ['mean']}

        if phase == 'val':
            self.gt_prim_sum = {x: 0 for x in list(range(opt.n_sem)) + ['mean']}
            self.prob_iou_exist_id = {x: [] for x in list(range(opt.n_sem)) + ['mean']}
            self.precision = {x: {y: [] for y in self.threshes} for x in list(range(opt.n_sem)) + ['mean']}
            self.recall = {x: {y: [] for y in self.threshes} for x in list(range(opt.n_sem)) + ['mean']}
            self.avg_precision = {x: {y: 0 for y in self.threshes} for x in list(range(opt.n_sem)) + ['mean']}

    def init_save(self, outputs_size):
        self.result_parts = np.zeros((outputs_size, opt.n_sem * opt.n_para))
        self.result_existence = np.zeros((outputs_size, 6))

    def accumulate(self, loss, loss_regress, loss_exist, iou, acc, outputs, existence, i_batch, phase):
        # print('===========================')
        # print(iou)
        # print(acc)
        # print(outputs)
        # print(i_batch, phase)
        # print('===========================')
        self.running_loss += loss.item()
        self.running_loss_regress += loss_regress.item()
        self.running_loss_exist += loss_exist.item()
        for key in self.running_iou.keys():
            if iou[key].item() >= 0:
                self.running_iou[key] += iou[key].item()
            else:
                self.inexistent[key] += 1###
            self.running_acc[key] += acc[key].item()

            if phase == 'val' or opt.demo is not None:
                if opt.exist and key != 'mean':
                    _, out_exist = outputs
                    self.prob_iou_exist_id[key].append((out_exist[0, key].item(), iou[key].item(),
                                                  existence[0, key].item(), i_batch))##
        # print(self.prob_iou_exist_id)
        # print('===========================')

    def compute_epoch(self, i_batch, phase):
        self.epoch_loss = self.running_loss / (i_batch + 1)
        self.epoch_loss_regress = self.running_loss_regress / (i_batch + 1)
        self.epoch_loss_exist = self.running_loss_exist / (i_batch + 1)
        for key in self.epoch_iou.keys():
            self.epoch_iou[key] = self.running_iou[key] / (i_batch + 1 - self.inexistent[key])
            self.epoch_acc[key] = self.running_acc[key] / (i_batch + 1)
        if phase == 'val' or opt.demo is not None:
            self.evaluate_precision_recall_ap()

    def update_best(self):
        self.lowest_val_loss = self.epoch_loss
        self.lowest_val_loss_regress = self.epoch_loss_regress
        self.lowest_val_loss_exist = self.epoch_loss_exist
        self.best_iou = copy.deepcopy(self.epoch_iou)
        self.best_acc = copy.deepcopy(self.epoch_acc)

    def is_best(self, i_batch, phase):
        self.compute_epoch(i_batch, phase)
        if phase == 'val' and self.epoch_loss < self.lowest_val_loss:
            self.update_best()
            return True
        else:
            return False

    def evaluate_precision_recall_ap(self):
        if sum(self.gt_prim_sum.values()) == 0:
            self.count_gt_prim_sum()
        self.compute_precision_recall()
        self.compute_average_precision()

    def count_gt_prim_sum(self):
        for key in self.prob_iou_exist_id.keys():
            if key != 'mean':
                for item in self.prob_iou_exist_id[key]:
                    self.gt_prim_sum[key] += item[2]

    def compute_precision_recall(self):
        # thresh = 0.4
        for thresh in self.threshes:
            for key in self.prob_iou_exist_id.keys():
                if key != 'mean':
                    res_list = self.prob_iou_exist_id[key]
                    res_list = sorted(res_list, key=lambda item:item[0], reverse=True)
                    tp = 0
                    fp = 0
                    for ins in res_list:
                        if ins[0] < 0:      # all negative in the last
                            break
                        if ins[1] >= thresh and ins[2]:     # true or false
                            tp += 1
                        else:
                            fp += 1
                        self.precision[key][thresh].append((tp / (tp + fp)))
                        self.recall[key][thresh].append((tp / self.gt_prim_sum[key]))

    def compute_average_precision(self):
        for thresh in self.threshes:
            for key in self.precision.keys():
                if key != 'mean':
                    ap = 0
                    for i in range(len(self.precision[key][thresh])):
                        if i == 0:
                            ap = self.precision[key][thresh][i] * self.recall[key][thresh][i]
                        else:
                            ap += self.precision[key][thresh][i] * (self.recall[key][thresh][i] - self.recall[key][thresh][i - 1])
                    self.avg_precision[key][thresh] = ap

    def save_file_mat_batch(self, i_batch, outputs):
        if opt.exist:
            out_parts, out_existence = outputs
            self.result_parts[i_batch, :] = out_parts.detach().cpu().numpy()
            self.result_existence[i_batch, :] = out_existence.detach().cpu().numpy()
        else:
            self.result_parts[i_batch, :] = outputs.detach().cpu().numpy()

    def save_file_mat(self, split):
        scipy.io.savemat('sem_result_{}.mat'.format(split),
                         {'parts': self.result_parts, 'existence': self.result_existence})

    def print_batch(self, epoch, i_batch):
        print('Epoch: {:4.0f} i_batch: {:4.0f} Loss_regress: {:.6f} Loss_exist: {:.6f} Loss: {:.6f} '
              'IOU: {:.6f} ACC: {:.6f}'.format(epoch, i_batch,
            self.running_loss_regress / (i_batch + 1), self.running_loss_exist / (i_batch + 1),
            self.running_loss / (i_batch + 1), self.running_iou['mean'] / (i_batch + 1 - self.inexistent['mean']),
            self.running_acc['mean'] / (i_batch + 1)))

    def print_epoch_begin(self, epoch):
        print('Epoch {}/{}'.format(epoch, opt.n_epochs) + '-' * 30)
        print('Time: %s' % datetime.datetime.now())

    def print_epoch_end(self, epoch, phase):
        print('{} Epoch: {:4.0f} Loss: {:.8f} epoch_iou: {:.6f} epoch_acc: {:.6f}'.format(
            phase, epoch, self.epoch_loss, self.epoch_iou['mean'], self.epoch_acc['mean']))
        print('lowest_val_regress: {:.8f} lowest_val_exist: {:.8f} '
              'lowest_val_loss: {:.8f} best_iou: {:.6f} best_acc: {:.6f}'.format(
            self.lowest_val_loss_regress, self.lowest_val_loss_exist, self.lowest_val_loss,
            self.best_iou['mean'], self.best_acc['mean']))
        time_elapsed = time.time() - self.since
        print('Time till now {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 60 // 60, time_elapsed // 60 % 60, time_elapsed % 60))

    def print_final(self):
        time_elapsed = time.time() - self.since
        print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 60 // 60, time_elapsed // 60 % 60, time_elapsed % 60))

    def plot(self, epoch, phase, is_best_tag=False):
        if opt.demo is None:
            self.plot_loss(epoch, phase)
            self.plot_eval(epoch, phase, 'mean_iou')
            self.plot_eval(epoch, phase, 'mean_acc')
            self.plot_iou_acc(epoch, phase, 'iou')
            self.plot_iou_acc(epoch, phase, 'acc')
        if (phase == 'val' and is_best_tag) or opt.demo is not None:
            self.plot_precision_recall_ap()
        self.vis.save(envs=[opt.env])

    def plot_loss(self, epoch, phase):
        self.plot_loss_one('Loss_all', epoch, phase)
        self.plot_loss_one('Loss_regress', epoch, phase)
        self.plot_loss_one('Loss_exist', epoch, phase)

    def plot_loss_one(self, win_name, epoch, phase):
        if win_name == 'Loss_all':
            self.vis.line(Y=torch.Tensor([self.epoch_loss]), X=torch.Tensor([epoch]),
                          win=win_name, env=opt.env,
                          opts=dict(title=win_name, xlabel='epoch', showlegend=True),
                          update='append',
                          name=phase)
            print(torch.Tensor([self.epoch_loss]), torch.Tensor([epoch]), phase)
        elif win_name == 'Loss_regress':
            self.vis.line(Y=torch.Tensor([self.epoch_loss_regress]), X=torch.Tensor([epoch]),
                          win=win_name, env=opt.env,
                          opts=dict(title=win_name, xlabel='epoch', showlegend=True),
                          update='append',
                          name=phase)
            print(torch.Tensor([self.epoch_loss_regress]), torch.Tensor([epoch]), phase)
        elif win_name == 'Loss_exist':
            self.vis.line(Y=torch.Tensor([self.epoch_loss_exist]), X=torch.Tensor([epoch]),
                          win=win_name, env=opt.env,
                          opts=dict(title=win_name, xlabel='epoch', showlegend=True),
                          update='append',
                          name=phase)
            print(torch.Tensor([self.epoch_loss_exist]), torch.Tensor([epoch]), phase)
        else:
            raise NotImplementedError

    def plot_eval(self, epoch, phase, metric):
        win_name = metric
        if metric == 'mean_iou':
            value = self.epoch_iou
        else:
            value = self.epoch_acc
        self.vis.line(Y=torch.Tensor([value['mean']]), X=torch.Tensor([epoch]),
                      win=win_name, env=opt.env,
                      opts=dict(title=win_name, xlabel='epoch', showlegend=True),
                      update='append',
                      name=phase)

    def plot_iou_acc(self, epoch, phase, metric):
        win_name = phase + '_' + metric
        if metric == 'iou':
            value = self.epoch_iou
        else:
            value = self.epoch_acc
        for key in value.keys():
            self.vis.line(Y=torch.Tensor([value[key]]), X=torch.Tensor([epoch]),
                          win=win_name, env=opt.env,
                          opts=dict(title=win_name, xlabel='epoch', showlegend=True),
                          update='append',
                          name=str(key))

    def plot_precision_recall_ap(self):
        self.plot_precision_recall()
        self.plot_ap_iou()
        self.plot_prim_count_histogram()

    def plot_precision_recall(self):
        for key in self.precision.keys():
            win_name = 'Precision_Recall_' + str(key)
            self.vis.close(win=win_name, env=opt.env)
            for thresh in self.threshes:
                if key != 'mean':
                    if len(self.precision[key][thresh]) == 0 or len(self.recall[key][thresh]) == 0:
                        continue
                    self.vis.line(Y=torch.Tensor(self.precision[key][thresh]), X=torch.Tensor(self.recall[key][thresh]),
                                  win=win_name, env=opt.env,
                                  opts=dict(title=win_name, ylabel='precision', xlabel='recall',
                                            ytickmin = 0., ytickmax=1.1, showlegend=True),
                                  update='append',
                                  name=str(thresh) + '%' + str(self.avg_precision[key][thresh]*100)[:4])

    def plot_ap_iou(self):
        win_name = 'AP_IOU'
        self.vis.close(win=win_name, env=opt.env)
        for key in self.avg_precision.keys():
            if key != 'mean':
                value = []
                for thresh in self.threshes:
                    value.append(self.avg_precision[key][thresh])
                self.vis.line(Y=torch.Tensor(value), X=torch.Tensor(self.threshes),
                              win=win_name, env=opt.env,
                              opts=dict(title=win_name, ylabel='avg precision', xlabel='threshold',
                                        ytickmin=0., ytickmax=1., showlegend=True),
                              update='append',
                              name=str(key))

    def plot_prim_count_histogram(self):
        win_name = 'Prim count'
        self.vis.bar(X=torch.Tensor(list(self.gt_prim_sum.values())[:6]), win=win_name, env=opt.env,
                     opts=dict(title=win_name, xlabel='prim_id',
                               rownames=['0', '1', '2', '3', '4', '5']))

