import time
import scipy.io
from lib.utils import StatisticsPrnn, Prim2VoxAccel, Eval
from lib.opts import *


def test_curve_one_epoch_init(epoch):
    global opt
    opt = get_opt()
    opt.model_epoch = epoch
    if int(opt.model_epoch) != -1:
        opt.pre_model = 'model_epoch_{}.pth'.format(opt.model_epoch)
        opt.init_source = opt.exp_prefix + '/' + opt.env + '_v{}_{}'.format(opt.test_version, opt.model_epoch)
    else:
        opt.init_source = opt.exp_prefix + '/' + opt.env + '_v{}_best'.format(opt.test_version)

    if not os.path.exists(opt.init_source):
        os.mkdir(opt.init_source)


def test_curve_one_epoch(epoch, since=None):
    prim_2_vox = Prim2VoxAccel()
    evaluator = Eval()
    for obj_class in opt.file_names['obj_classes']:
        test_curve_one_epoch_one_class(epoch, obj_class, evaluator, prim_2_vox, since)


def test_curve_one_epoch_one_class(epoch, obj_class, evaluator, prim_2_vox, since):
    if since is None:
        since = time.time()
    statis = StatisticsPrnn()
    test_version = opt.test_version
    if 'str' in opt.metric or 'hausdorff' in opt.metric or \
            'box2d' in opt.metric or 'ins' in opt.metric or 'recall' in opt.metric:
        if opt.tul:
            dist, acc, cls_acc, box2d_acc, box2d_iou, c_pr, b_pr, b_iou, loss_str = evaluator.cal_dist_one_class_tul(obj_class)
        else:
            dist, acc, cls_acc, box2d_acc, box2d_iou, c_pr, b_pr, b_iou, loss_str = evaluator.cal_dist_one_class(obj_class)
        if 'ins' in opt.metric:
            for ins_i in range(len(loss_str[5])):
                statis.plot_iou_one(ins_i, loss_str[5][ins_i], str(epoch),
                                    'Loss_r ins test_version v{}'.format(test_version))
        if 'str' in opt.metric:
            statis.plot_iou_one(epoch, loss_str[0], obj_class, 'Loss_x str test_version v{}'.format(test_version))
            statis.plot_iou_one(epoch, loss_str[1], obj_class, 'Loss_r str test_version v{}'.format(test_version))
            if opt.out_r == 'class':
                statis.plot_iou_one(epoch, loss_str[2], obj_class,
                                    'Loss_r_theta str test_version v{}'.format(test_version))
                statis.plot_iou_one(epoch, loss_str[3], obj_class,
                                    'Loss_r_axis str test_version v{}'.format(test_version))
            if opt.loss_c is not None or opt.len_adjust:
                statis.plot_iou_one(epoch, loss_str[4], obj_class,
                                    'Loss_c str test_version v{}'.format(test_version))
        if 'recall' in opt.metric:
            for t in opt.h_thresh:
                statis.plot_iou_one(epoch, acc[t], obj_class, 'Hausdorff recall {} test_version v{}'.format(
                    t, test_version))
        if 'hausdorff' in opt.metric:
            # dist, acc, cls_acc, _, _, _, _, _, _ = evaluator.cal_dist_one_class(obj_class)
            statis.plot_iou_one(epoch, dist, obj_class, 'Hausdorff dist test_version v{}'.format(test_version))
            for t in opt.h_thresh:
                statis.plot_iou_one(epoch, acc[t], obj_class, 'Hausdorff acc {} test_version v{}'.format(
                    t, test_version))
            statis.plot_iou_one(epoch, cls_acc[0], obj_class, 'Semantic acc test_version v{}'.format(test_version))
    if opt.nms_thresh > 0:
        dist_file = opt.init_source + '/dist_{}_{}.mat'.format(obj_class, opt.eval_mode)
        nms_removed = scipy.io.loadmat(dist_file)['nms_removed']
    else:
        nms_removed = None
    if 'iou' in opt.metric:
        if opt.obj_class[:6] == '3dprnn':
            opt.file_names['voxel'] = 'primset_sem/modelnet_all.mat'
        if 'aabb' in opt.metric and 'Myvoxel' in opt.file_names['voxel'] and 'nyu' not in opt.obj_class:
            opt.file_names['voxel'] = opt.file_names['voxel'][:-4] + '_aabb.mat'
        if 'existing' in opt.metric:
            iou_file = opt.init_source + '/iou_{}.mat'.format(obj_class)
            iou = scipy.io.loadmat(iou_file)['mean_iou'][0, 0]
        else:
            vox = prim_2_vox.prim_to_vox_one_class_accel(obj_class, nms_removed=nms_removed, aabb=('aabb' in opt.metric))
            iou = evaluator.cal_iou_one_class(obj_class, pred=vox)
        statis.plot_iou_one(epoch, iou, obj_class, 'IoU test_version v{}'.format(test_version))
    if 'shape' in opt.metric:
        if opt.obj_class[:6] == '3dprnn':
            opt.file_names['voxel'] = 'primset_sem/Myvoxel_{}.mat'.format(opt.part)
        if 'aabb' in opt.metric and 'Myvoxel' in opt.file_names['voxel']:
            opt.file_names['voxel'] = opt.file_names['voxel'][:-4] + '_aabb.mat'
        if 'existing' in opt.metric:
            iou_file = opt.init_source + '/iou_{}_shape.mat'.format(obj_class)
            iou = scipy.io.loadmat(iou_file)['mean_iou'][0, 0]
        else:
            vox = prim_2_vox.prim_to_vox_one_class_accel(obj_class, nms_removed=nms_removed, aabb=('aabb' in opt.metric))
            iou = evaluator.cal_iou_one_class(obj_class, pred=vox, metric='shape')
        statis.plot_iou_one(epoch, iou, obj_class, 'IoU-Shape v{}'.format(test_version))
    # acc = evaluator.cal_acc_one_class(obj_class)
    # statis.plot_torch_loss()
    time_elapsed = time.time() - since
    print('Epoch {} {} till now {:.0f}h {:.0f}m {:.0f}s'.format(
        epoch, obj_class, time_elapsed // 60 // 60, time_elapsed // 60 % 60, time_elapsed % 60))
