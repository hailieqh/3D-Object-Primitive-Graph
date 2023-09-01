import torch
import logging
import time
from distutils.version import LooseVersion
from sacred import Experiment
from easydict import EasyDict as edict
from logging.config import dictConfig
import scipy.io

# from lib.utils import random_init, check_para_correctness, create_logger
from lib.prnn import PRNN
from lib.utils import Eval, Prim2Vox, Prim2VoxAccel, StatisticsPrnn, random_init
from lib.datasets import NNCompute, SaveGTt7
from lib.utils.prnn_utils import *
from lib.opts import *
from lib.test import *
import pdb


if __name__ == '__main__':
    global opt
    opt = get_args()
    set_opt(opt)
    print(opt)
    random_init(0)
    load_primset_to_t7_format(opt.data_dir, opt.obj_class) ## set opt.mean_std
    if not os.path.exists(opt.exp_prefix):
        os.mkdir(opt.exp_prefix)
    if not os.path.exists(opt.exp_dir):
        os.mkdir(opt.exp_dir)
    if not os.path.exists('out'):
        os.mkdir('out')
    if not os.path.exists(opt.init_source):
        os.mkdir(opt.init_source)
    if opt.eval_mode is not None:
        test_offline()
    elif opt.test_curve:
        test_through_iou_curve()
    elif opt.nn_curve:
        nn_iou_curve()
    elif opt.test_through:
        test_through_one_model()
    elif opt.save_w_vector:
        prim_predictor = PRNN(cfg=None, logger=None)
        prim_predictor.save_train_val_test_w()
        nn_computer = NNCompute(inverse=opt.inverse)
        nn_computer.compute_nn_all_class()
    elif opt.prim2vox:
        prim_2_vox = Prim2VoxAccel()
        prim_2_vox.prim_to_vox_all_class()
    elif opt.cal_iou:
        evaluator = Eval()
        evaluator.cal_iou_all_class()
    else:
        prim_predictor = PRNN(cfg=None, logger=None)
        if opt.stage == 'ssign':
            if opt.demo is None:
                prim_predictor.train_ssign()
            else:
                prim_predictor.test_ssign()
        elif opt.plot_val:
            prim_predictor.val()
        elif opt.demo is None:
            if opt.train_nn_init:
                opt.save_w_vector = True
                prim_predictor1 = PRNN(cfg=None, logger=None)
                prim_predictor1.save_train_val_test_w() ## resnet fc random
                del prim_predictor1
                nn_computer = NNCompute(inverse=opt.inverse)
                nn_computer.compute_nn_one_class('train')
                nn_computer.compute_nn_one_class('val')
            prim_predictor.train()
        else:
            prim_predictor.test()
