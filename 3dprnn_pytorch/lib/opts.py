import argparse
import os
import torch
import numpy as np


def get_args():
    proj_dir = os.path.abspath('.')
    parser = argparse.ArgumentParser(description='3D Semantic Object Part Detection')

    parser.add_argument('--stage',         default=None,              help='bbox or prnn')
    parser.add_argument('--use_gpu',       action='store_true',       help='Whether or not to use a GPU')
    parser.add_argument('--GPU',           default=None,              help='GPU ID, if None no GPU')
    parser.add_argument('--snapshot',      default=5,                 help='Model checkpoint')
    parser.add_argument('--n_epochs',      default=None,              help='Number of training epochs')
    parser.add_argument('--train_batch',   default=1,                 help='Training mini-batch size')
    parser.add_argument('--valid_batch',   default=1,                 help='Validation mini-batch size')
    parser.add_argument('--test_batch',    default=1,                 help='Test mini-batch size')
    parser.add_argument('--num_workers',   default=4,                 help='')

    parser.add_argument('--lr',            default=1e-4,              help='Learning rate')
    parser.add_argument('--eps',           default=None,              help='Epsilon')
    parser.add_argument('--momentum',      default=None,              help='Momentum')
    parser.add_argument('--weight_decay',  default=None,              help='Weight decay')
    parser.add_argument('--alpha',         default=None,              help='Alpha')
    parser.add_argument('--step_size',     default=None,              help='Learning rate decay step size')
    parser.add_argument('--gamma',         default=None,              help='Learning rate decay gamma')

    parser.add_argument('--data_dir',      default=None,              help='Data directory')
    parser.add_argument('--resnet_dir',    default=None,              help='Pretrained resnet model directory')
    parser.add_argument('--exp_prefix',    default=None,              help='Experiment directory')
    parser.add_argument('--exp_dir',       default=None,              help='Model directory')
    parser.add_argument('--pre_dir',       default=None,              help='Pretrained model directory')
    parser.add_argument('--param_dir',     default=None,              help='Pretrained model params directory')
    parser.add_argument('--pre_model',     default='best_model.pth',  help='Pretrained model name')
    parser.add_argument('--model_name',    default='resnet18',        help='Defined model name')
    parser.add_argument('--demo',          default=None,              help='Train or val or test split in testing mode')

    parser.add_argument('--visual_data',   action='store_true',       help='Visualize data before and after processing')
    parser.add_argument('--visdom',        action='store_true',       help='Use Visdom')
    parser.add_argument('--env',           default='env',             help='Name of visdom environment')

    parser.add_argument('--m_dir_mode',    default=0,                 help='Model dir')
    parser.add_argument('--m_param_mode',  default=0,                 help='Model parameters')
    parser.add_argument('--expand',        default=0,                 help='Expansion of gt bbox')
    parser.add_argument('--n_sem',         default=6,                 help='Number of semantic parts in an object')
    parser.add_argument('--n_para',        default=6,                 help='Number of parameters for a semantic part')
    parser.add_argument('--v_size',        default=0,                 help='Volume size of ground truth objects')
    parser.add_argument('--exist',         action='store_true',       help='Enable existence of semantic parts in an object')
    parser.add_argument('--reg_partial',   action='store_true',       help='Only compute reg loss of existent parts')
    parser.add_argument('--max_pool',      action='store_true',       help='Replace avgpool by maxpool in resnet')
    parser.add_argument('--canonical',     action='store_true',       help='Canonical view')

    # parser.add_argument('--input_size',    default=3,                 help='Number of input dimension')
    parser.add_argument('--hid_size',      default=400,               help='Number of hidden units in lstms')
    parser.add_argument('--max_len',       default=100,               help='Max sequence length')
    parser.add_argument('--hid_layer',     default=3,                 help='Number of hidden layers')
    # parser.add_argument('--num_passes',    default=1,                 help='Number of passes')
    parser.add_argument('--con_size',      default=None,              help='Condition size')
    parser.add_argument('--pred_len',      default=40,                help='Prediction length')
    # parser.add_argument('--depth_con',     action='store_true',       help='Conditioned on depth')
    parser.add_argument('--obj_class',     default=None,              help='Object class')
    parser.add_argument('--c_sigma',       default=0,                 help='Count if sigma is too small')
    parser.add_argument('--save_nn',       action='store_true',       help='Save nearest neighbor for test depth maps')
    parser.add_argument('--cal_iou',       action='store_true',       help='Calculate mean iou')
    parser.add_argument('--init_torch',    action='store_true',       help='Init for torch')
    parser.add_argument('--save_w_vector', action='store_true',       help='Save depth codes for PyTorch model')
    parser.add_argument('--test_through',  action='store_true',       help='Test the saved model on all classes')
    parser.add_argument('--init_source',   default='init_ours',       help='Init source: given or ours')
    parser.add_argument('--train_val',     action='store_true',       help='Train the model on train and val splits')
    parser.add_argument('--composition',   default=None,              help='Data composition: None, balance, chair, table, night_stand')
    parser.add_argument('--machine',       default='mac_h',           help='Run code on which machine: mac_h, ai')
    parser.add_argument('--model_epoch',   default=-1,                help='Pretrained model epoch id')
    parser.add_argument('--test_gap',      default=-1,                help='Test depth image gap')
    parser.add_argument('--prim2vox',      action='store_true',       help='Translate primitives to voxel')
    parser.add_argument('--test_curve',    action='store_true',       help='Test and draw iou curves on all classes')
    parser.add_argument('--nn_curve',      action='store_true',       help='Test and draw nn iou curves on all classes')
    parser.add_argument('--save_gt_t7',    action='store_true',       help='Save train and val gt from .t7')
    parser.add_argument('--gt_init',       action='store_true',       help='Init test with gt, instead of nn')
    parser.add_argument('--gt_in',         action='store_true',       help='GT input for each lstm step')
    parser.add_argument('--test_version',  default=-1,                help='Test version ID for current model')
    parser.add_argument('--shift_res',     default=0.,                help='Shift result before translating to voxel')
    parser.add_argument('--intervals',     default=None,              help='Train validation intervals')
    parser.add_argument('--file_names',    default=None,              help='File names of inputs and outputs')
    parser.add_argument('--sigma_min',     default=0.,                help='Min of sigma prediction')
    parser.add_argument('--sigma_reg',     default=0.,                help='Sigma regularization weight')
    parser.add_argument('--sigma_share',   action='store_true',       help='Share sigma for all components')
    parser.add_argument('--sigma_abs',     action='store_true',       help='Replace exp with abs')
    # parser.add_argument('--rgb_con',       action='store_true',       help='Conditioned on rgb')
    parser.add_argument('--shuffle',       action='store_true',       help='Shuffle data')
    parser.add_argument('--fix_encoder',   action='store_true',       help='Fix resnet encoder')
    parser.add_argument('--model_split',   action='store_true',       help='Split by model')
    parser.add_argument('--gmm_weight',    default=1.,                help='Gmm loss weight')
    parser.add_argument('--r_weight',      default=1.,                help='Rotation loss weight')
    parser.add_argument('--box2d_weight',  default=1.,                help='Box2d loss weight')
    parser.add_argument('--c_weight',      default=1.,                help='Semantic label loss weight')
    parser.add_argument('--n_component',   default=20,                help='Number of Gaussian component')
    parser.add_argument('--part',          default='all',             help='Which semantic part to train on')
    parser.add_argument('--loss_c',        default=None,              help='Loss for output class label')
    parser.add_argument('--bbox_con',      default='None',            help='In: pre_ or ora_ and all or bbox or exist')
    parser.add_argument('--filter',        action='store_true',       help='Filter through bbox')
    parser.add_argument('--f_thresh',      default=0.,                help='Filter through bbox threshold')
    parser.add_argument('--bbox_loss',     default='None',            help='Bbox constraint on primitives')
    parser.add_argument('--encoder',       default=None,              help='Encoder: depth, resnet, or hg')
    parser.add_argument('--fpn',           action='store_true',       help='Use FPN')
    parser.add_argument('--box2d_source',  default=None,              help='Box2d source: oracle or prediction')
    parser.add_argument('--box2d_size',    default=0,                 help='Box2d local feature encoding size')
    parser.add_argument('--box2d_pos',     default='0',               help='Box2d encoding position type')
    parser.add_argument('--box2d_en',      default=None,              help='Box2d position encoding')
    parser.add_argument('--cut_data',      action='store_true',       help='Cut down training data from 3dprnn')
    parser.add_argument('--fix_bn',        action='store_true',       help='Fix mean and var')
    parser.add_argument('--loss_y',        default='gmm',             help='Loss for scale and translation')
    parser.add_argument('--loss_e',        default=None,              help='Loss for stop sign')
    parser.add_argument('--loss_r',        default=None,              help='Loss for rotation')
    parser.add_argument('--loss_box2d',    default=None,              help='Loss for primitive 2D bounding box')
    parser.add_argument('--out_r',         default='3dprnn',          help='Loss for rotation')
    parser.add_argument('--step_in',       default=None,              help='Input for each step of LSTM')
    parser.add_argument('--step_in_ep',    default=2000,              help='Step in epoch')
    parser.add_argument('--train_nn_init', action='store_true',       help='Init train with nearest neighbor model')
    parser.add_argument('--zero_in',       action='store_true',       help='Zero inputs')
    parser.add_argument('--gt_perturb',    default=0.,                help='Add gt perturbation for training')
    parser.add_argument('--continue_t',    action='store_true',       help='Continue to test')
    parser.add_argument('--no_rand',       action='store_true',       help='No randomness for stop sign')
    parser.add_argument('--stop_thresh',   default=0.5,               help='Threshold for stop sign')
    parser.add_argument('--plot_val',      action='store_true',       help='Plot validation loss curve')
    parser.add_argument('--stop_sign_w',   action='store_true',       help='Add weight to stop sign to balance 0 and 1')
    parser.add_argument('--stop_sign_reg', action='store_true',       help='Regress stop sign directly')
    parser.add_argument('--feat_bn',       action='store_true',       help='Add BN after resnet feature')
    parser.add_argument('--ss_l2',         default=None,              help='Load stop sign prediction from L2')
    parser.add_argument('--dim_e',         default=1,                 help='Dimension of stop sign')
    # parser.add_argument('--xyz',           action='store_true',       help='True if output a primitive for each step')
    parser.add_argument('--init_in',       default=None,              help='Initialization input of lstm')
    parser.add_argument('--eval_mode',     default=None,              help='Evaluation mode')
    parser.add_argument('--revise_mode',   default=None,              help='Prediction revision mode')
    parser.add_argument('--add_fcs',       default=None,              help='Add more fc layers after resnet')
    parser.add_argument('--metric',        default='',                help='Evaluation metric')
    parser.add_argument('--prim_norm',     default=None,              help='Normalize the primitive wrt p_i 0-n')
    parser.add_argument('--global_denorm', action='store_true',       help='Denormalize s, t, r globally')
    parser.add_argument('--val_loss',      action='store_true',       help='Validation loss consistent with testing except for init')

    # semantic starmap
    parser.add_argument('--dataset',       default='pix3d',           help='Dataset choice: bed | ***')
    parser.add_argument('--nFeats',        default=256,               help='Number of features in the hourglass')
    parser.add_argument('--nStack',        default=1,                 help='Number of hourglasses to stack')
    parser.add_argument('--nModules',      default=1,                 help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--input_res',     default=256,               help='Input image resolution')
    parser.add_argument('--output_res',    default=64,                help='Output heatmap resolution')
    parser.add_argument('--hmGauss',       default=1,                 help='Heatmap gaussian size')
    parser.add_argument('--nClasses',      default=6,                 help='Number of keypoint classes for bed or other things')
    parser.add_argument('--PCPthr',        default=1.5,               help='Threshold for PCP or PCK torso')
    parser.add_argument('--sem_map',       default=None,              help='Use semantic keypoint heatmap as attention')
    parser.add_argument('--tmp',           action='store_true',       help='Tmp flag for debugging')
    parser.add_argument('--debug',         default=None,              help='Test a module')

    parser.add_argument('--refine',        default=None,              help='Prim refinement module')
    parser.add_argument('--n_hid_g',       default=8,                 help='Number of hidden units in graph')
    parser.add_argument('--n_head_g',      default=8,                 help='Number of attention heads')
    parser.add_argument('--alpha_g',       default=0.2,               help='Alpha for leaky_relu')
    parser.add_argument('--drop_g',        default=0.6,               help='Dropout rate for graph')
    parser.add_argument('--n_out_g',       default=32,                help='Graph output size for each node')
    parser.add_argument('--graph_pre',     default=None,              help='Graph pretrain')
    parser.add_argument('--feature_scale', default='None',            help='Feature scaling strategy')
    parser.add_argument('--test_loss',     default=None,              help='Calculate test loss')
    parser.add_argument('--adj',           default='fc',              help='Adjacent matrix init')
    parser.add_argument('--n_graph',       default=2,                 help='Number of graph layers')
    parser.add_argument('--fix_prnn',      default='yes',             help='Fix pretrained prnn')
    parser.add_argument('--n_att',         default=0,                 help='Additional multi-head attention layer')
    parser.add_argument('--residual',      action='store_true',       help='Refine residual')
    parser.add_argument('--xyz',           default=1,                 help='Prim xyz as one step')
    parser.add_argument('--gpu_base',      action='store_true',       help='For memory')
    parser.add_argument('--check',         default='None',            help='Check data by printing')
    parser.add_argument('--sem_reg',       action='store_true',       help='Different regressors for different sem parts')
    parser.add_argument('--refine_ep',     default='300',             help='Refine epoch of lstm')
    parser.add_argument('--scale_box2d',   default=1.,                help='Scale of box2d, enlarge or not')
    parser.add_argument('--lr_r',          default=1e-4,              help='Learning rate for rotation regression')
    parser.add_argument('--reg',           default=None,              help='Regressor for scale, trans and rot')
    parser.add_argument('--mean_std',      default=None,              help='Mean and std of prim params')
    parser.add_argument('--norm_rot',      default=None,              help='Normalize rotation wrt to theta or all')
    parser.add_argument('--norm_st',       default=None,              help='Normalize scale and trans together')

    parser.add_argument('--f_dim',         default=256,    type=int, help='Dimension for graph node feature')
    parser.add_argument('--m_dim',         default=0,      type=int, help='Dimension for middle regression layer')
    parser.add_argument('--intra',         default=0,      type=int, help='Intra semantic graph message propagation times')
    parser.add_argument('--inter',         default=0,      type=int, help='Inter semantic graph message propagation times')
    parser.add_argument('--mp',            default=1,      type=int, help='Message passing times')
    parser.add_argument('--stack_mp',      default=1,      type=int, help='Message passing times among graph stacks')
    parser.add_argument('--refine_mp',     default=1,      type=int, help='Refine module message passing times')
    parser.add_argument('--node',          default='str',            help='Graph refine targets')
    parser.add_argument('--refine_node',   default='str',            help='Graph refine targets')
    parser.add_argument('--update',        default='str_cls',        help='Update after graph mp')
    parser.add_argument('--graph',         default='design',         help='Graph type')
    parser.add_argument('--embed_type',    default='0',              help='Node embedding type')
    parser.add_argument('--refine_embed',  default='0',              help='Node embedding type in refine mp')
    parser.add_argument('--mes_type',      default='0',              help='Message type')
    parser.add_argument('--out_refine',    default='None',           help='Output graph refinement')
    parser.add_argument('--embed_fuse',    default=0,      type=int, help='Fuse embeddings')

    parser.add_argument('--re_lstm',       default=0,      type=int, help='Retrain LSTM from epoch re_lstm')
    parser.add_argument('--reg_init',      default='None',           help='Regress init for lstm')
    parser.add_argument('--match',         default='sequence',       help='Compute loss with matching method')
    parser.add_argument('--stack_learn',   default='None',           help='Stacked learning split a/b')
    parser.add_argument('--len_source',    default='None',           help='Source of length: gt, pred, stack')
    parser.add_argument('--len_adjust',    action='store_true',      help='Adjust length after graph')
    parser.add_argument('--len_max',       default=0,       type=int, help='Assign max length')
    parser.add_argument('--cc_weight',     default=1.,    type=float, help='Inexistent part label loss weight')
    parser.add_argument('--bg_lstm',       action='store_true',      help='Back ground class for lstm')
    parser.add_argument('--sem_select',    default='pred',           help='Semantic selection for mlps source')
    parser.add_argument('--sem_box2d',     action='store_true',      help='Use different mlps for different sem')
    parser.add_argument('--sem_graph',     action='store_true',      help='Different MLPs for parts')
    parser.add_argument('--sem_final',     action='store_true',      help='Different MLPs for parts for final output')

    parser.add_argument('--inverse',       action='store_true',      help='Inverse sequence order for lstm proposal')
    parser.add_argument('--proposal',      default=None,             help='Save proposals from lstm')
    parser.add_argument('--bi_lstm',       default=None,             help='Bidirectional lstm proposal')
    parser.add_argument('--bi_match',      default=None, type=int,   help='Match lstm proposal with gt twice')
    parser.add_argument('--drop_test',     default=None, type=float, help='Drop part of pred')
    parser.add_argument('--nms_thresh',    default=0.,   type=float, help='NMS threshold for hausdorff dist')
    parser.add_argument('--h_thresh',      default=None,             help='Hausdorff threshold')
    parser.add_argument('--no_sem',        action='store_true',      help='No semantics')
    parser.add_argument('--full_model',    action='store_true',      help='Full model instead of half')
    parser.add_argument('--pqnet',         action='store_true',      help='No need to complete the right half for pqnet')
    parser.add_argument('--tul',           action='store_true',      help='Evaluate hausdorff metrics for tulsiani, with --pqnet for no sym')
    parser.add_argument('--len_dir',       default='length_all',     help='Length file dir')
    parser.add_argument('--step_start',    default=1,    type=int,   help='Step offset of the first cls pred')
    parser.add_argument('--lstm_prop',     default=None,             help='Proposal from gmm model')
    parser.add_argument('--graph_xyz',     default=None, type=int,   help='Prim xyz as one step in graph')
    parser.add_argument('--extra_len',     default=0,    type=int,   help='Extra proposal length')
    parser.add_argument('--vis_node',      default='admin',          help='Visdom node')
    parser.add_argument('--agnostic',      action='store_true',      help='Class-agnostic testing')
    parser.add_argument('--lstm2',         action='store_true',      help='NMS for 2 lstm results')
    # parser.add_argument('--prop_xyz',      default=None, type=int,   help='Prim xyz as one step in proposal generation')

    parser.add_argument('--partial_graph', default='',               help='Partial graph construction rule')
    parser.add_argument('--faster',        default='',               help='Use 2D proposals from faster-rcnn')
    parser.add_argument('--n_prop',        default=30,   type=int,   help='Proposal maximum number')
    parser.add_argument('--one2many',      action='store_true',      help='Match pred to gt')
    parser.add_argument('--recall_tmp',    action='store_true',      help='Compute our LSTM*2 recall')
    args = parser.parse_args()

    # train
    args.machine = 'ai'
    args.fix_bn = True
    if 'all' not in args.obj_class:
        args.shuffle = True
    args.fix_encoder = True
    if args.loss_y == 'gmm':
        pass
        # args.xyz = 1
        # args.out_r = '3dprnn'
    else:
        args.no_rand = True
        args.stop_thresh = 0.5
    # args.reg_init = 'share'
    args.norm_rot = 'theta'
    args.norm_st = 'st'
    # test
    if args.reg_init != 'None':
        args.step_start = 0


    n_sem_all = {'chair': 6, 'table': 4, 'sofa': 4, 'desk': 5, 'bed': 5,
                 'nyuchair': 6, 'nyutable': 4, 'nyusofa': 4,
                 '3dprnnchair': 6, '3dprnntable': 4, '3dprnnnight_stand': 4,
                 '3dprnnnyuchair': 6, '3dprnnnyutable': 4, '3dprnnnyunight_stand': 4,
                 'all': 1, '3dprnnall': 1}
    args.n_sem = n_sem_all[args.obj_class]

    if args.proposal == 'save':
        args.valid_batch = 1

    if 'recall' in args.metric:
        args.h_thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    else:
        args.h_thresh = [0.1, 0.2, 0.3]

    if args.m_dim == 0:
        args.m_dim = args.f_dim

    if args.full_model:
        args.len_dir = 'length_all_full'

    args.param_dir = proj_dir + '/../../data/model_params/{}'.format(args.obj_class)
    args.resnet_dir = proj_dir + '/../../data/model_params/resnet'
    if args.stage == 'bbox':
        args.input_res = 224
        args.n_epochs = 150
        args.lr = 1e-1
        args.weight_decay = 1e-4
        args.momentum = 0.9
        args.step_size = 30
        args.gamma = 1e-1
        args.pre_dir = args.param_dir + '/model_stage_one_no_existence/best_model.pth'
    elif args.stage == 'ssign':
        args.n_epochs = 300
        args.eps = 1e-6
        args.pre_dir = args.resnet_dir + '/resnet18.pth'
    elif args.stage == 'kp':
        args.n_epochs = 150
        args.lr = 2.5e-4
        args.weight_decay = 0.
        args.momentum = 0.
        args.alpha = 0.99
        args.eps = 1e-8
        args.nStack = int(args.nStack)
    elif args.stage == 'prnn':
        args.eps = 1e-6
        if args.encoder == 'resnet':
            args.pre_dir = args.resnet_dir + '/resnet18.pth'
            # args.pre_dir = args.param_dir + '/model_stage_one_bbox/best_model.pth'
        elif args.encoder == 'hg':
            args.pre_dir = args.param_dir + '/model_stage_one_hg_stack_{}/best_model.pth'.format(args.nStack)
        refine_dir_all = {}

        #############################################################
        refine_dir_all['chair'] = 'exp_p02c01_2'
        refine_dir_all['chair_inv'] = 'exp_p02c01_3'
        if args.no_sem:
            refine_dir_all['chair'] = 'exp_p02c01_22'
            refine_dir_all['chair_inv'] = 'exp_p02c01_33'
            args.n_sem = 1
        if args.loss_box2d is None:
            refine_dir_all['chair'] = 'exp_p02c01_222'
            refine_dir_all['chair_inv'] = 'exp_p02c01_333'

        if args.bi_lstm == 'online':
            args.pre_dir = {}
            args.pre_dir['forward'] = args.param_dir + \
                                      '/{}/model_epoch_{}.pth'.format(
                                          refine_dir_all[args.obj_class], args.refine_ep)
            args.pre_dir['backward'] = args.param_dir + \
                                       '/{}/model_epoch_{}.pth'.format(
                                           refine_dir_all[args.obj_class+'_inv'], args.refine_ep)
        refine_dir = refine_dir_all[args.obj_class]
        if args.inverse:
            refine_dir = refine_dir_all[args.obj_class+'_inv']
        if args.refine is not None:
            args.pre_dir = args.param_dir + \
                           '/{}/model_epoch_{}.pth'.format(refine_dir, args.refine_ep)
        if args.re_lstm != 0:
            args.pre_dir = args.param_dir + \
                           '/{}/model_epoch_{}.pth'.format(refine_dir, abs(args.re_lstm))
        if args.encoder == 'resnet' or args.encoder == 'hg':
            args.con_size = 256
            args.test_gap = 1
            args.n_epochs = 400
            args.v_size = 32
            if args.obj_class[:6] == '3dprnn':
                args.v_size = 30
        elif args.encoder in ['depth', 'depth_new']:
            args.con_size = 32
            args.test_gap = 10
            args.n_epochs = 300
            args.v_size = 30
            if args.encoder == 'depth_new':
                args.test_gap = 1
    else:
        raise NotImplementedError

    args.n_epochs = int(args.n_epochs)
    args.snapshot = int(args.snapshot)
    args.train_batch = int(args.train_batch)
    args.valid_batch = int(args.valid_batch)
    args.test_batch = int(args.test_batch)
    args.num_workers = int(args.num_workers)
    args.input_res = int(args.input_res)
    args.output_res = int(args.output_res)

    args.expand = float(args.expand)
    args.n_sem = int(args.n_sem)
    args.n_para = int(args.n_para)      # + int(args.exist)
    args.v_size = int(args.v_size)
    args.m_dir_mode = int(args.m_dir_mode)
    args.m_param_mode = int(args.m_param_mode)
    args.nClasses = args.n_sem

    args.test_gap = int(args.test_gap)
    args.shift_res = float(args.shift_res)
    args.sigma_min = float(args.sigma_min)
    args.sigma_reg = float(args.sigma_reg)
    args.gmm_weight = float(args.gmm_weight)
    args.box2d_weight = float(args.box2d_weight)
    args.r_weight = float(args.r_weight)
    args.c_weight = float(args.c_weight)
    args.n_component = int(args.n_component)
    args.f_thresh = float(args.f_thresh)
    args.hid_layer = int(args.hid_layer)
    args.hid_size = int(args.hid_size)
    args.lr = float(args.lr)
    args.lr_r = float(args.lr_r)
    args.step_in_ep = int(args.step_in_ep)
    args.gt_perturb = float(args.gt_perturb)
    args.stop_thresh = float(args.stop_thresh)
    args.box2d_size = int(args.box2d_size)

    args.n_hid_g = int(args.n_hid_g)
    args.n_head_g = int(args.n_head_g)
    args.n_out_g = int(args.n_out_g)
    args.drop_g = float(args.drop_g)
    args.n_graph = int(args.n_graph)
    args.n_att = int(args.n_att)
    args.xyz = int(args.xyz)
    args.scale_box2d = float(args.scale_box2d)
    if args.weight_decay is None:
        args.weight_decay = 0.
    else:
        args.weight_decay = float(args.weight_decay)

    if args.demo is not None:
        args.train_batch = args.valid_batch

    args.use_gpu = args.GPU is not None and torch.cuda.is_available()

    args.exp_prefix = proj_dir + '/exp' + args.stage
    args.exp_dir = args.exp_prefix + '/exp_' + args.env
    args.init_source = args.env
    if int(args.model_epoch) != -1:
        args.pre_model = 'model_epoch_{}.pth'.format(args.model_epoch)
        args.init_source = args.exp_prefix + '/' + args.env + '_v{}_{}'.format(args.test_version, args.model_epoch)
    else:
        args.init_source = args.exp_prefix + '/' + args.env + '_v{}_best'.format(args.test_version)

    args.intervals = {'train': [3335, 4805, 5555], 'val': [1110, 1600, 1850]}
    if args.encoder in ['resnet', 'hg', 'depth_new']:
        args.data_dir = proj_dir + '/../../data/all_classes/{}'.format(args.obj_class)
        args.file_names = {'depth': {'train': 'depth_mn_train_all_tr_pth.mat',
                                     'val': 'depth_mn_train_all_val_pth.mat'},
                           'primset': 'primset_sem/Myprimset_{}.mat'.format(args.part),
                           'voxel': 'primset_sem/Myvoxel_{}.mat'.format(args.part),
                           'nums': [994, 361, 3839, 700, 68, 1947, 1870, 47, 243],
                           'intervals': [994, 1355, 5194, 5894, 5962, 7909, 9779, 9826, 10069],
                           'obj_classes': [args.obj_class]
                           }
    elif args.encoder == 'depth':
        args.data_dir = proj_dir + '/../../data/3dprnn'
        args.file_names = {'depth': {'train': 'depth_mn_train_all_tr_pth.mat',
                                     'val': 'depth_mn_train_all_val_pth.mat'},
                           'primset': 'primset_sem/Myprimset_{}.mat'.format(args.part),
                           'norm_label': {'train': 'prim_rnn_batch_nz_all_tr_py.pth',
                                          'val': 'prim_rnn_batch_nz_all_val_py.pth'},
                           'mean_std': 'sample_generation/test_NNfeat_mn_{}.mat'.format(args.obj_class),
                           'obj_classes': ['chair', 'table', 'night_stand'],
                           'intervals': {'train': [3335, 4805, 5555], 'val': [1110, 1600, 1850]},
                           }
    else:
        raise NotImplementedError


    if 'all' in args.obj_class:
        args.data_dir = proj_dir + '/../../data/all_classes'
        args.len_dir = '{}/length_all'.format(args.obj_class)
        if args.obj_class == 'all':
            args.file_names['obj_classes'] = ['chair', 'table', 'sofa']
        elif args.obj_class == '3dprnnall':
            args.file_names['obj_classes'] = ['3dprnnchair', '3dprnntable', '3dprnnnight_stand']
        args.file_names['img_nums'] = {}

    if args.stage in ['prnn', 'bbox', 'ssign']:
        args.canonical = True
    args.visdom = True
    args.model_split = True
    if args.loss_e is None:
        args.dim_e = 0

    return args


def set_opt(value):
    global opt
    opt = value


def get_opt():
    return opt
