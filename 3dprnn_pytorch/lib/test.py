import time
import os
from lib.prnn import PRNN
from lib.utils import Eval, Prim2Vox, Prim2VoxAccel, StatisticsPrnn, random_init
from lib.datasets import NNCompute, SaveGTt7
from lib.utils.prnn_utils import *
from lib.utils.test_utils import *
from lib.opts import *
import pdb


def nn_iou_curve():
    global opt
    opt = get_opt()
    for epoch in range(0, 501, opt.snapshot):
        print('Epoch:', epoch)
        opt.model_epoch = epoch
        if int(opt.model_epoch) != -1:
            opt.pre_model = 'model_epoch_{}.pth'.format(opt.model_epoch)
            opt.init_source = opt.exp_prefix + '/' + opt.env + '_v{}_{}'.format(opt.test_version, opt.model_epoch)
        else:
            opt.init_source = opt.exp_prefix + '/' + opt.env + '_v{}_best'.format(opt.test_version)

        if not os.path.exists(opt.init_source):
            os.mkdir(opt.init_source)

        evaluator = Eval()
        for obj_class in opt.file_names['obj_classes']:
            statis = StatisticsPrnn()
            iou = evaluator.cal_iou_one_class_by_vox(obj_class)
            statis.plot_iou_one(epoch, iou, obj_class, 'NN IoU test_version v{}'.format(opt.test_version))


def test_through_iou_curve():
    global opt
    opt = get_opt()
    since = time.time()
    if 'ts' in opt.test_version:
        start_epoch = int(opt.model_epoch)
        end_epoch = start_epoch + opt.snapshot + 1
    elif opt.continue_t:
        start_epoch = int(opt.model_epoch)
        end_epoch = 501
        if start_epoch > 1000:
            end_epoch = start_epoch + 1
    else:
        start_epoch = 10
        end_epoch = 501
    for epoch in range(start_epoch, end_epoch, opt.snapshot): #range(500,-1,-20): #range(0, 501, 20):
        test_curve_one_epoch_init(epoch)

        if 'nn' in opt.metric:
            if not opt.gt_init:
                opt.save_w_vector = True
                prim_predictor1 = PRNN(cfg=None, logger=None)
                prim_predictor1.save_train_val_test_w()
                del prim_predictor1
                nn_computer = NNCompute(inverse=opt.inverse)
                nn_computer.compute_nn_all_class()
            else:
                # if opt.rgb_con:
                if opt.encoder == 'resnet' or opt.encoder == 'hg':
                    raise NotImplementedError
                nn_computer = NNCompute(inverse=opt.inverse)
                nn_computer.init_from_gt_all_class()

        opt.save_w_vector = False
        opt.demo = 'all'

        if 'pred' in opt.metric:
            prim_predictor2 = PRNN(cfg=None, logger=None)
            prim_predictor2.test()
            del prim_predictor2

        test_curve_one_epoch(epoch, since)


def test_offline():
    global opt
    opt = get_opt()
    since = time.time()
    if int(opt.model_epoch) != -1:
        opt.pre_model = 'model_epoch_{}.pth'.format(opt.model_epoch)
        opt.init_source = opt.exp_prefix + '/' + opt.env + '_v{}_{}'.format(opt.test_version, opt.model_epoch)
    else:
        opt.init_source = opt.exp_prefix + '/' + opt.env + '_v{}_best'.format(opt.test_version)

    evaluator = Eval()
    evaluator.cal_iou_all_class()
    time_elapsed = time.time() - since
    print('Epoch {} till now {:.0f}h {:.0f}m {:.0f}s'.format(
        opt.model_epoch, time_elapsed // 60 // 60, time_elapsed // 60 % 60, time_elapsed % 60))


def test_through_one_model():
    global opt
    opt = get_opt()
    opt.save_w_vector = True
    prim_predictor1 = PRNN(cfg=None, logger=None)
    prim_predictor1.save_train_val_test_w()
    del prim_predictor1
    nn_computer = NNCompute(inverse=opt.inverse)
    nn_computer.compute_nn_all_class()

    opt.save_w_vector = False
    opt.demo = 'all'
    # opt.depth_con = True
    prim_predictor2 = PRNN(cfg=None, logger=None)
    prim_predictor2.test()
    del prim_predictor2

    prim_2_vox = Prim2VoxAccel()
    evaluator = Eval()
    for cls in opt.file_names['obj_classes']:
        since = time.time()
        vox = prim_2_vox.prim_to_vox_one_class_accel(cls)
        evaluator.cal_iou_one_class(cls, pred=vox)
        time_elapsed = time.time() - since
        print('{} complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            cls, time_elapsed // 60 // 60, time_elapsed // 60 % 60, time_elapsed % 60))
