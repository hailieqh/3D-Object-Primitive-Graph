# Scripts for Chair in Pix3D

# Step 1: estimate the number of primitives for the object in an image with stacked learning
#sequence length 
#length_{gt, pred}_{test_chair, train, val}.mat
CUDA_VISIBLE_DEVICES=0 python main.py --obj_class chair --stage ssign --machine ai --encoder resnet --fix_bn --shuffle \
--fix_encoder --input_res 224 --train_batch 16 --valid_batch 1 --xyz 3 --norm_rot theta --norm_st st --loss_e l2 \
--lr 1e-4 --m_dir_mode 1 --m_param_mode 0 --GPU 0 --env 00000_0 2>&1|tee train00000_0.log
CUDA_VISIBLE_DEVICES=0 python main.py --obj_class chair --stage ssign --machine ai --encoder resnet --fix_bn --shuffle \
--fix_encoder --input_res 224 --train_batch 16 --valid_batch 1 --xyz 3 --stack_learn a --norm_rot theta --norm_st st --loss_e l2 \
--lr 1e-4 --m_dir_mode 1 --m_param_mode 0 --GPU 0 --env 00000_2 2>&1|tee train00000_2.log
CUDA_VISIBLE_DEVICES=0 python main.py --obj_class chair --stage ssign --machine ai --encoder resnet --fix_bn --shuffle \
--fix_encoder --input_res 224 --train_batch 16 --valid_batch 1 --xyz 3 --stack_learn b --norm_rot theta --norm_st st --loss_e l2 \
--lr 1e-4 --m_dir_mode 1 --m_param_mode 0 --GPU 0 --env 00000_3 2>&1|tee train00000_3.log
#test 
CUDA_VISIBLE_DEVICES=0 python main.py --obj_class chair --stage ssign --machine ai --encoder resnet --xyz 3 \
--norm_rot theta --norm_st st --loss_e l2 --fix_bn --shuffle --fix_encoder --input_res 224 --lr 1e-4 --m_dir_mode 1 --m_param_mode 0 \
--GPU 0 --env 00000_0 --demo all 
CUDA_VISIBLE_DEVICES=0 python main.py --obj_class chair --stage ssign --machine ai --encoder resnet --xyz 3 --stack_learn a \
--norm_rot theta --norm_st st --loss_e l2 --fix_bn --shuffle --fix_encoder --input_res 224 --lr 1e-4 --m_dir_mode 1 --m_param_mode 0 \
--GPU 0 --env 00000_2 --demo all 
CUDA_VISIBLE_DEVICES=0 python main.py --obj_class chair --stage ssign --machine ai --encoder resnet --xyz 3 --stack_learn b \
--norm_rot theta --norm_st st --loss_e l2 --fix_bn --shuffle --fix_encoder --input_res 224 --lr 1e-4 --m_dir_mode 1 --m_param_mode 0 \
--GPU 0 --env 00000_3 --demo all

# Step 2: train proposal lstm models
#lstm, forward, box2d, cls, sem_select gt
CUDA_VISIBLE_DEVICES=0 python main.py --encoder resnet --obj_class chair --stage prnn --machine ai --fix_bn --shuffle --train_val \
--fix_encoder --input_res 224 --train_batch 16 --valid_batch 30 --xyz 3 \
--reg_init share --hid_layer 2 --out_r theta --norm_rot theta --norm_st st --sem_reg --sem_select gt \
--fpn --box2d_source pred_detach --box2d_size 32 --box2d_pos 0 --box2d_en dense_norm \
--loss_y l1c --loss_r l1c --loss_c nllc --loss_box2d sl1c --gmm_weight 10 --box2d_weight 10 --scale_box2d 1.1 \
--lr 1e-4 --GPU 1 --env p02c01_2 2>&1|tee train_p02c01_2.log 
#--inverse
CUDA_VISIBLE_DEVICES=0- python main.py --encoder resnet --obj_class chair --stage prnn --machine ai --fix_bn --shuffle --train_val \
--fix_encoder --input_res 224 --train_batch 16 --valid_batch 30 --xyz 3 --inverse \
--reg_init share --hid_layer 2 --out_r theta --norm_rot theta --norm_st st --sem_reg --sem_select gt \
--fpn --box2d_source pred_detach --box2d_size 32 --box2d_pos 0 --box2d_en dense_norm \
--loss_y l1c --loss_r l1c --loss_c nllc --loss_box2d sl1c --gmm_weight 10 --box2d_weight 10 --scale_box2d 1.1 \
--lr 1e-4 --GPU 1 --env p02c01_3 2>&1|tee train_p02c01_3.log 


# Step3: train graph network
#graph
CUDA_VISIBLE_DEVICES=0 python main.py --encoder resnet --obj_class chair --stage prnn --train_val \
--input_res 224 --train_batch 16 --valid_batch 30 --reg_init share --graph_xyz 3 --xyz 3 --out_r theta \
--bi_lstm online --bi_match 2 --match set --len_source stack --len_adjust --step_in detach --step_in_ep 0 \
--hid_layer 2 --sem_reg --sem_final --sem_select gt --fix_prnn yes \
--refine_ep 20 --f_dim 1024 --node h_cot --embed_type 1 --mes_type 0 --mp 1 --update str_cls --graph base \
--fpn --box2d_source pred_detach --box2d_size 32 --box2d_pos 0 --box2d_en dense_norm \
--loss_y l1c --loss_r l1c --loss_c nllc --loss_box2d sl1c --gmm_weight 10 --box2d_weight 10 --scale_box2d 1.1 \
--lr 1e-4 --GPU 1 --env p02c02_1 2>&1|tee train_p02c02_1.log
#test
CUDA_VISIBLE_DEVICES=0 python main.py --encoder resnet --obj_class chair --test_version 1 --test_curve \
--stage prnn --input_res 224 --GPU 2 --ss_l2 pred --bi_lstm online --bi_match 2 --len_source stack --len_adjust \
--reg_init share --graph_xyz 3 --xyz 3 --out_r theta --hid_layer 2 --sem_reg --sem_final --sem_select gt --fix_prnn yes \
--refine_ep 20 --f_dim 1024 --node h_cot --embed_type 1 --mes_type 0 --mp 1 --update str_cls --graph base \
--fpn --box2d_source pred_detach --box2d_size 32 --box2d_pos 0 --box2d_en dense_norm \
--loss_y l1c --loss_r l1c --loss_c nllc --loss_box2d sl1c --gmm_weight 10 --box2d_weight 10 --scale_box2d 1.1 \
--metric pred_hausdorff_iou --drop_test 1 --env p02c02_1 --continue_t --model_epoch 400


