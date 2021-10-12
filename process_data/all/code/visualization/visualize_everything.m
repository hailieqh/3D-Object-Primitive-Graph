% visualize gt primitive
clear all;
addpath('./Voxel Plotter/');

global cls part phase sem_flag save_visual exp voxel_scale local_root nms
global server_root img_dir color data_root p300_root save_pred_dir n_sem len_adjust demo_rot
cls =  'chair';
part = 'all';
phase = 'test';
sem_flag = 0; %n_sem=6;
len_adjust = 1;
nms = 1;
save_visual = 1;
demo_rot = -1;
exp = 'd02n02_11_v1_2000';
voxel_scale = 32; %30 or 32
color = {
    [1 1 0],...
    [1 0 1],...
    [0 1 0],...
    [0 1 1],...
    [0 0.1 1],...
    [1 0.4 0.4],...
    [1 0.6 0],...
    [0 0.74 1],...
    [0.62 0.12 0.94],...
    [0.4 0.4 0.4],...
    [1 0.9 0.5],...
    [0.4 0.9 0],...
    [0 0 0]
    };
cls_all = {'chair', 'bed', 'bookcase', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe'};
if 1
    local_root = '/Users/heqian/Research/projects/primitive-based_3d';
    server_root = '/Users/heqian/Research/1112/model_split/primitive-based_3d';
    data_root = '/Users/heqian/Research/1112/model_split';%p300';
    p300_root = '/Users/heqian/Research/p300';
    img_dir = [data_root '/data/all_classes/' cls '/images_crop_object/'];
    save_pred_dir = '/Users/heqian/Research/projects';
else
    data_root = 'C:\Users\HQBB\Desktop\final_prim';
    p300_root = 'C:\Users\HQBB\Desktop\final_prim';
    save_pred_dir = 'C:\Users\HQBB\Desktop\final_prim';
    img_dir = [data_root '\data\all_classes\' cls '\images_crop_object\'];
end

if strcmp(cls, 'chair') || strcmp(cls, '3dprnnchair')
    n_sem = 6;
elseif strcmp(cls, 'table') || strcmp(cls, '3dprnntable')
    n_sem = 4;
elseif strcmp(cls, 'sofa') || strcmp(cls, '3dprnnnight_stand')
    n_sem = 4;
end

usage_all = {'downsample', 'edit', 'pred'};
usage = 'pred';
if strcmp(usage, 'downsample')
    for i = 2:1:numel(cls_all)
        cls = cls_all{i};
        visualize_downsample();
    end
elseif strcmp(usage, 'voxelize')
%     cls = 'chair';
    sem_flag = 1;
    visualize_voxelize();
elseif strcmp(usage, 'edit')
%     cls = 'chair';
    sem_flag = 1;
    finished = 1;
    start = 116;
    align_choices = {'remained', 'deleted', 'whole'};
    align = align_choices{3};
    visualize_edit(finished, align, start);
elseif strcmp(usage, 'pred')
    full_model = 1; %1 for faster rcnn and pqnet
    visualize_pred(full_model);
elseif strcmp(usage, 'demo')
    full_model = 0;
    visualize_demo(full_model);
elseif strcmp(usage, 'demo_rot')
    full_model = 0;
    visualize_demo_final_rot(full_model);
elseif strcmp(usage, 'demo_refine')
    full_model = 0;
    visualize_demo_refine(full_model);
elseif strcmp(usage, 'match')
    visualize_vox_prim_match();
elseif strcmp(usage, 'edit_3dprnn')
%     cls = 'chair';
    sem_flag = 1;
    finished = 1;
    start = 1;
    visualize_3dprnn_edit(finished, start);
end

%% visualize 3dprnn edit
function [out] = visualize_3dprnn_edit(finished, start)
global local_root cls voxel_scale color sem_flag part
root = [local_root '/process_data/all'];
output_dir = [root '/output/' cls];
visual_save_dir = [root '/visual/edit_3dprnn/' cls];
mkdir(visual_save_dir);
voxel_modelnet = load([output_dir '/primset_sem/modelnet_all.mat']);
voxel_modelnet = voxel_modelnet.voxTile;
% voxTile = voxel_modelnet;
% save_model_dir = [output_dir '/primset_sem/modelnet_all.mat'];
load([output_dir '/primset_sem/Myprimset_' part '.mat'], 'primset');
for i = start:1:size(voxel_modelnet, 1)
%     disp(i);
    voxel_down_i = reshape(voxel_modelnet(i,:,:,:),voxel_scale,voxel_scale,voxel_scale);
%     if (strcmp(cls, '3dprnntable') && i ~= 17)
%         voxel_down_i = rot90(voxel_down_i);
%         voxTile(i,:,:,:) = voxel_down_i;
%         save(save_model_dir, 'voxTile');
%     end
    I = figure(1);
    subplot(1,3,1);plot_voxel(voxel_down_i, voxel_scale);title('align');
    prim_num = size(primset{i}.ori, 1);
    for j = 1:prim_num
        if sem_flag
            sem_id = primset{i}.cls(j);
        else
            sem_id = j;
        end
        prim_r = primset{i}.ori(j,:);
        plot_a_prim_twice_voxelize(prim_r, color, sem_id, voxel_scale, 3);
        prim_r = primset{i}.sym(j,:);
        plot_a_prim_twice_voxelize(prim_r, color, sem_id, voxel_scale, 3);
    end
    saveas(I, sprintf([visual_save_dir '/%d.jpg'], i));
    close(I);
end
out = 1;
end


%% visualize 3dprnn voxel match
function [out] = visualize_vox_prim_match()
global local_root voxel_scale cls part sem_flag color
root = [local_root '/process_data/all'];
output_dir = [root '/output/' cls];
visual_save_dir = [root '/visual/model_match/' cls];
mkdir(visual_save_dir);
voxel_modelnet = load([output_dir '/primset_sem/modelnet_all.mat']);
voxel_modelnet = voxel_modelnet.voxTile;
% voxel_voxelized = voxel_modelnet;
voxel_voxelized = load([output_dir '/primset_sem/Myvoxel_all.mat']);
voxel_voxelized = voxel_voxelized.voxTile;
load([output_dir '/primset_sem/Myprimset_' part '.mat'], 'primset');
for i = 1:1:size(voxel_modelnet, 1)
%     disp(i);
    voxel_down_i = reshape(voxel_modelnet(i,:,:,:),voxel_scale,voxel_scale,voxel_scale);
    voxel_voxelized_i = reshape(voxel_voxelized(i,:,:,:),voxel_scale,voxel_scale,voxel_scale);
    I = figure(1);
    subplot(1,2,1);plot_voxel(voxel_down_i, voxel_scale);title('before');
    subplot(1,2,2);plot_voxel(voxel_voxelized_i, voxel_scale);title('after');
    prim_num = size(primset{i}.ori, 1);
    for j = 1:prim_num
        if sem_flag
            sem_id = primset{i}.cls(j);
        else
            sem_id = j;
        end
        prim_r = primset{i}.ori(j,:);
        plot_a_prim_twice_voxelize(prim_r, color, sem_id, voxel_scale, 2);
        prim_r = primset{i}.sym(j,:);
        plot_a_prim_twice_voxelize(prim_r, color, sem_id, voxel_scale, 2);
    end
    saveas(I, sprintf([visual_save_dir '/%d.jpg'], i));
end
out = 1;
end

%% visualize prim editing
function [out] = visualize_edit(finished, align, start)
global local_root cls voxel_scale color sem_flag
root = [local_root '/process_data/all'];
output_dir = [root '/output/' cls];
edit_dir = [output_dir '/mat'];
visual_save_dir = [root '/visual/edit/' cls];
mkdir(visual_save_dir);
voxel_down = load([output_dir '/downsampled_voxels_' cls '.mat']);
voxel_down = voxel_down.voxTile;
primset = load([edit_dir '/Myprimset.mat']);
primset = primset.primset;
if ~finished
    for i = start:1:6000
        disp(i);
        vox_gt = reshape(voxel_down(i,:,:,:),voxel_scale,voxel_scale,voxel_scale);
        I = figure(1);
        plot_voxel_twice(vox_gt, voxel_scale);

%         myvox = load([edit_dir '/Myvox' num2str(i) '.mat']);
%         myvox = myvox.new_vox;
        myprim = load([edit_dir '/Myprim' num2str(i) '.mat']);
        myprim = myprim.prim_save;
        prim_num = size(myprim, 1);
        for j = 1:prim_num
            prim_r = myprim(j, 1:20);
            if sem_flag
                sem_id = myprim(j, 22);
            else
                sem_id = j;
            end
            if strcmp(align, 'remained')
                vox_gt = myvox.remained(j,:,:,:);
            elseif strcmp(align, 'deleted')
                vox_gt = myvox.deleted(j,:,:,:);
            end
            vox_gt = reshape(vox_gt,voxel_scale,voxel_scale,voxel_scale);
            subplot(1,3,2);
            plot_voxel(vox_gt, voxel_scale);
            plot_a_prim_twice_edit(prim_r, color, sem_id, voxel_scale);
        end
    end
else
    for i = start:1:6000
        disp(i);
        vox_gt = reshape(voxel_down(i,:,:,:),voxel_scale,voxel_scale,voxel_scale);
        I = figure(1);
        plot_voxel_twice(vox_gt, voxel_scale);
        
        prim_num = size(primset{i}.ori, 1);
        for j = 1:prim_num
            if sem_flag
                sem_id = primset{i}.cls(j);
            else
                sem_id = j;
            end
            prim_r = primset{i}.ori(j,:);
            plot_a_prim_twice_edit(prim_r, color, sem_id, voxel_scale);
            prim_r = primset{i}.sym(j,:);
            plot_a_prim_twice_edit(prim_r, color, sem_id, voxel_scale);
        end
        saveas(I, sprintf([visual_save_dir '/%d.jpg'], i));
    end
end
out = true;
end

%% visualize voxelization
function [out] = visualize_voxelize()
global local_root voxel_scale cls part sem_flag color
root = [local_root '/process_data/all'];
output_dir = [root '/output/' cls];
visual_save_dir = [root '/visual/voxelization/' cls];
mkdir(visual_save_dir);
voxel_down = load([output_dir '/downsampled_voxels_' cls '.mat']);
voxel_down = voxel_down.voxTile;
voxel_voxelized = load([output_dir '/primset_sem/Myvoxel_all.mat']);
voxel_voxelized = voxel_voxelized.voxTile;
load([output_dir '/primset_sem/Myprimset_' part '.mat'], 'primset');
for i = 1:1:size(voxel_down, 1)
%     disp(i);
    voxel_down_i = reshape(voxel_down(i,:,:,:),voxel_scale,voxel_scale,voxel_scale);
    voxel_voxelized_i = reshape(voxel_voxelized(i,:,:,:),voxel_scale,voxel_scale,voxel_scale);
    I = figure(1);
    subplot(1,2,1);plot_voxel(voxel_down_i, voxel_scale);title('before');
    subplot(1,2,2);plot_voxel(voxel_voxelized_i, voxel_scale);title('after');
    prim_num = size(primset{i}.ori, 1);
    for j = 1:prim_num
        if sem_flag
            sem_id = primset{i}.cls(j);
        else
            sem_id = j;
        end
        prim_r = primset{i}.ori(j,:);
        plot_a_prim_twice_voxelize(prim_r, color, sem_id, voxel_scale, 2);
        prim_r = primset{i}.sym(j,:);
        plot_a_prim_twice_voxelize(prim_r, color, sem_id, voxel_scale, 2);
    end
    saveas(I, sprintf([visual_save_dir '/%d.jpg'], i));
end
out = true;
end

%% visualize downsampling
function [out] = visualize_downsample()
global local_root voxel_scale cls
root = [local_root '/process_data/all'];
voxel_ori_dir = [root '/input/model/'];
output_dir = [root '/output/' cls];
visual_save_dir = [root '/visual/downsample/' cls];
mkdir(visual_save_dir);
voxel_down = load([output_dir '/downsampled_voxels_' cls '.mat']);
voxel_down = voxel_down.voxTile;
fileID = fopen([output_dir '/voxels_dir_' cls '.txt'],'r');
tline = fgetl(fileID);
voxel_names{1} = tline;
count = 1;
while ischar(tline)
    tline = fgetl(fileID);
    count = count + 1;
    voxel_names{count} = tline;
end
fclose(fileID);
voxel_names(end) = [];
for i = 1:1:size(voxel_names, 2)
%     disp(i);
    voxel_name = voxel_names{i};
    voxel = load([voxel_ori_dir voxel_name]);
    voxel = voxel.voxel;
    bound_lower = sum(sum(voxel(1,:,:)))+sum(sum(voxel(:,1,:)))+sum(sum(voxel(:,:,1)));
    bound_upper = sum(sum(voxel(128,:,:)))+sum(sum(voxel(:,128,:)))+sum(sum(voxel(:,:,128)));
    if bound_lower == 0 && bound_upper == 0
        disp(i)
    end
    voxel_down_i = reshape(voxel_down(i,:,:,:),voxel_scale,voxel_scale,voxel_scale);
    I = figure(1);
    subplot(1,2,1);plot_voxel(voxel, 128);title('before');
    subplot(1,2,2);plot_voxel(voxel_down_i, voxel_scale);title('after');
    saveas(I, sprintf([visual_save_dir '/%d.jpg'], i));
end
out = true;
end

%% visualize prediction
function [out] = visualize_pred(full_model)
global cls part phase sem_flag save_visual exp voxel_scale local_root server_root img_dir color data_root p300_root
load([data_root '/data/all_classes/' cls '/primset_sem/Myprimset_' part '.mat'], 'primset');
% vox_data = load([data_root '/data/all_classes/' cls '/primset_sem/prim_to_voxel_' cls '_' part '.mat']);
vox_data = load([data_root '/data/all_classes/' cls '/primset_sem/Myvoxel_' part '.mat']);
vox_data = vox_data.voxTile;%%%%%to be updated
res = load([p300_root '/3dprnn_pytorch/expprnn/' exp '/test_res_mn_' cls '.mat']);
iou = load([p300_root '/3dprnn_pytorch/expprnn/' exp '/dist_' cls '_None_aabb.mat']);  %iou_' cls '.mat'
% hist(iou.iou_chair);
fileID = fopen([data_root '/data/all_classes/' cls '/voxeltxt/' phase '.txt'],'r');
img_vox_id = load([data_root '/data/all_classes/' cls '/img_voxel_idxs.mat']);
for i = 1:1:6000%1:numel(primset)
    disp(i);
    I = figure(1);
     set(gcf, 'WindowStyle', 'docked');
    img_name = fgetl(fileID);
    %img_name = '0952.jpg';
    [prim_num_gt] = plot_image_and_match_id(img_name, img_dir, img_vox_id);
    vox_gt = reshape(vox_data(prim_num_gt,:,:,:),voxel_scale,voxel_scale,voxel_scale);
    %subplot(2,3,1);plot_voxel(vox_gt, voxel_scale);
    %subplot(2,3,2);plot_voxel(vox_gt, voxel_scale);title('gt prim & vox');
    h4 = subplot(2,3,4);cla(h4);
    h5 = subplot(2,3,5);cla(h5);
    %ll = draw_res(i, res, color, voxel_scale, 1, iou, sem_flag, full_model);
    ll = draw_res(i, res, color, voxel_scale, 4, iou, sem_flag, full_model);
    if ~isempty(primset{prim_num_gt})
        size_prim = size(primset{prim_num_gt}.ori,1);
    else
        out = save_img(i, save_visual, I, cls, exp);
        close(I);
        continue
    end
    for j = 1:size_prim
        if sem_flag
            sem_id = primset{prim_num_gt}.cls(j);
        else
            sem_id = j;
        end
        if sem_id > size(color, 2)
            sem_id = sem_id - size(color, 2);
        end
        prim_r = primset{prim_num_gt}.ori(j,:);
        plot_a_prim_twice_pred(prim_r, color, sem_id, voxel_scale);
        prim_r = primset{prim_num_gt}.sym(j,:);
        plot_a_prim_twice_pred(prim_r, color, sem_id, voxel_scale);
    end
    %keyboard
    out = save_img(i, save_visual, I, cls, exp, iou);
    close(I);
end
fclose(fileID);
end

%% visualize demo
function [out] = visualize_demo(full_model)
global cls part phase sem_flag save_visual exp voxel_scale local_root server_root 
global color data_root p300_root n_sem len_adjust demo_rot
res = load([p300_root '/3dprnn_pytorch/expprnn/' exp '/test_res_mn_' cls '.mat']);
iou = load([p300_root '/3dprnn_pytorch/expprnn/' exp '/dist_' cls '_None.mat']);  %iou_' cls '.mat'
% hist(iou.iou_chair);
img_vox_id = load([data_root '/data/all_classes/' cls '/img_voxel_idxs.mat']);
img_dir = '/Users/heqian/Research/projects/primitive-based_3d/process_data/all/input/img/chair/';
demo_save_dir = '/Users/heqian/Research/projects/primitive-based_3d/process_data/all/output/demo/';
demo_save_dir = [demo_save_dir exp '/'];
mkdir(demo_save_dir);

% instance info
%p02c01_2_v1_20
%p02c02_1_v1_400//p02c02_0_v1_65
img_name = '1397.jpg';
img_bbox = [790, 387, 2407, 3301];
i = 198;
img_id = str2num(img_name(1:4));
disp(img_id);
% load image
im = im2double(imread([img_dir img_name]));
im = crop_bbox(im, img_bbox);

% instance prediction
res_x = res.x;
res_cls = res.cls;
res_box2d = res.box2d;
res_prim = res_x((i-1)*4+1:(i-1)*4+2, :);
res_rot = res_x((i-1)*4+3, :);
res_sym = res_x(i*4, :);
res_bbox = res_box2d((i-1)*4+1:(i-1)*4+4, :);
stop_idx = find(res_prim(1,:) == 0,1,'first');

% res_bbox = [[0.11367742 0.14363512 0.2743834  0.21088775 0.13703613 0.24760541 ...
%   0.24084692 0.21762325 0.         0.         0.        ];
%  [0.1696953  0.07503279 0.06794973 0.38291579 0.24017267 0.58249837 ...
%   0.47899261 0.54125214 0.         0.         0.        ];
%  [0.25674963 0.34869003 0.647587   0.81897771 0.31211391 0.71492893 ...
%   0.69515467 0.31633827 0.         0.         0.        ];
%  [0.38048768 0.45920691 0.46957532 0.70937866 0.68317258 0.69300854 ...
%   0.59152704 0.92163372 0.         0.         0.        ]];
% res_bbox = [[ 21.86595494,  65.43602763,  69.33991414, 135.3807703 ];
%        [ 33.82886508,  33.76076934, 101.86981143, 161.23672606];
%        [ 84.34172433,  31.464986  , 208.177469  , 164.7316484 ];
%        [ 65.06279116, 145.18058325, 266.8381036 , 253.50689737];
%        [ 30.92288958,  95.01279817,  89.01687402, 242.00822223];
%        [ 76.12929744, 214.4453648 , 231.19571641, 251.11464959];
%        [ 73.44650338, 176.69935154, 224.19407838, 214.04031932];
%        [ 59.07026007, 203.86790487,  91.82569298, 330.08543922]]';
%p=min(h,w)/320,pad = 0;
% res_c = [4     1     1     2     3     5     5     3];

I = figure(1);
%set(gcf, 'WindowStyle', 'docked');
hold off;cla;
add=1;
start = add*3+1-3;
tol_cnt = add;
for res_row = start:3:stop_idx-3
    prim_rot = res_rot(res_row:res_row+2);
    prim_r = [res_prim(1, res_row:res_row+2) res_prim(2, res_row:res_row+2) prim_rot];
    sym_r = res_sym(res_row:res_row+2); 
    bbox = res_bbox(:, res_row)';
    %bbox = res_bbox(:, tol_cnt);
    bbox = enlarge_bbox(bbox);
    if sem_flag
        sem_id = res_cls(i, res_row) + 1;
        %sem_id = res_c(tol_cnt);
        if sem_id > n_sem && len_adjust
            continue
        end
    else
        sem_id = res_cls(i, res_row) + 1;
        if sem_id > 1 && len_adjust
            continue
        end
        sem_id = tol_cnt;
    end
    %sem_id = add;

    vertices = prim9_to_vertices(prim_r, voxel_scale, 0);
    plot_a_prim_alone(vertices, color, sem_id, voxel_scale);
    
    saveas(I, sprintf([demo_save_dir '%d_%d.jpg'], i, tol_cnt));
    
    [h, w, ~] = size(im);
    p = max(h,w);
    pad = abs(h-w)/2;
    if w > h
        x_min = floor(bbox(1) * p) + 1;
        y_min = floor(bbox(2) * p - pad) + 1;
        x_max = floor(bbox(3) * p) + 1;
        y_max = floor(bbox(4) * p - pad) + 1;
    else
        x_min = floor(bbox(1) * p - pad) + 1;
        y_min = floor(bbox(2) * p) + 1;
        x_max = floor(bbox(3) * p - pad) + 1;
        y_max = floor(bbox(4) * p) + 1;
    end
    width = x_max - x_min;
    height = y_max - y_min;
    bbox_i = [x_min y_min width height];
    %bbox_i = [100, 300, 200, 500];%[x y width height];
    im_i = insertShape(im, 'FilledRectangle', bbox_i, 'Opacity', 0.3, 'Color', color{sem_id});
    imwrite(im_i, sprintf([demo_save_dir img_name(1:4) '_%d.jpg'], tol_cnt));
    
    tol_cnt = tol_cnt + 1;
end

imwrite(im, [demo_save_dir img_name(1:4) '.jpg']);
% plot full prediction
close(I);
I = figure(1);
%set(gcf, 'WindowStyle', 'docked');
ll = draw_res_demo(i, res, color, voxel_scale, iou, sem_flag, full_model);
saveas(I, sprintf([demo_save_dir '%d.jpg'], i));
end

%% visualize demo refinement
function [out] = visualize_demo_refine(full_model)
global cls part phase sem_flag save_visual voxel_scale local_root server_root 
global color data_root p300_root n_sem len_adjust
img_dir = '/Users/heqian/Research/projects/primitive-based_3d/process_data/all/input/img/chair/';
demo_save_dir = '/Users/heqian/Research/projects/primitive-based_3d/process_data/all/output/demo/';

exp1 = 'p02c01_2_v1_20';
exp2 = 'p02c02_1_v1_400';
demo_save_dir = [demo_save_dir exp2 '/refine/'];
mkdir(demo_save_dir);

res1 = load([p300_root '/3dprnn_pytorch/expprnn/' exp1 '/test_res_mn_' cls '.mat']);
iou1 = load([p300_root '/3dprnn_pytorch/expprnn/' exp1 '/dist_' cls '_None.mat']);  %iou_' cls '.mat'

res2 = load([p300_root '/3dprnn_pytorch/expprnn/' exp2 '/test_res_mn_' cls '.mat']);
iou2 = load([p300_root '/3dprnn_pytorch/expprnn/' exp2 '/dist_' cls '_None.mat']);  %iou_' cls '.mat'

% instance info
i = 198;
disp(i);
interval = 180;

for inter_i = 0:1:interval
    I = figure(1);
    %set(gcf, 'WindowStyle', 'docked');
    ll = draw_res_demo_refine(i, inter_i/interval, res1, res2, color, voxel_scale, sem_flag, full_model);
    saveas(I, sprintf([demo_save_dir '%08d.jpg'], inter_i));
    close(I);
end
end

%% visualize demo final rot
function [out] = visualize_demo_final_rot(full_model)
global cls part phase sem_flag save_visual exp voxel_scale local_root server_root 
global color data_root p300_root n_sem len_adjust demo_rot
res = load([p300_root '/3dprnn_pytorch/expprnn/' exp '/test_res_mn_' cls '.mat']);
iou = load([p300_root '/3dprnn_pytorch/expprnn/' exp '/dist_' cls '_None.mat']);  %iou_' cls '.mat'
img_dir = '/Users/heqian/Research/projects/primitive-based_3d/process_data/all/input/img/chair/';
demo_save_dir = '/Users/heqian/Research/projects/primitive-based_3d/process_data/all/output/demo/';
demo_save_dir = [demo_save_dir exp '/rot/'];
mkdir(demo_save_dir);

% instance info
%p02c02_1_v1_400
i = 198;%nms for draw_res_demo(remove final prim)
disp(i);
% instance prediction
%res_x = res.x;
%res_cls = res.cls;
%res_box2d = res.box2d;
%res_prim = res_x((i-1)*4+1:(i-1)*4+2, :);
%res_rot = res_x((i-1)*4+3, :);
%res_sym = res_x(i*4, :);
%res_bbox = res_box2d((i-1)*4+1:(i-1)*4+4, :);
%stop_idx = find(res_prim(1,:) == 0,1,'first');

for rot_i = 1:1:720
    demo_rot = (rot_i-1) * pi / 360;
    I = figure(1);
    %set(gcf, 'WindowStyle', 'docked');
    ll = draw_res_demo(i, res, color, voxel_scale, iou, sem_flag, full_model);
    saveas(I, sprintf([demo_save_dir '%08d.jpg'], rot_i));
    close(I);
end
end

%% enlarge bbox
function [bbox] = enlarge_bbox(bbox)
rf = 75;
pad = 37 / 224;
bbox(1) = max(0, bbox(1) - pad);
bbox(2) = max(0, bbox(2) - pad);
bbox(3) = min(1, bbox(3) + pad);
bbox(4) = min(1, bbox(4) + pad);
end

%% crop bbox
function [image] = crop_bbox(image, bbox)
thresh = 0.1;
[h, w, ~] = size(image);
x_min = bbox(1);
y_min = bbox(2);
x_max = bbox(3);
y_max = bbox(4);
x_min = max(0, x_min - (x_max - x_min) * thresh) + 1;
x_max = min(w, x_max + (x_max - x_min) * thresh) + 1;
y_min = max(0, y_min - (y_max - y_min) * thresh) + 1;
y_max = min(h, y_max + (y_max - y_min) * thresh) + 1;
image = image(floor(y_min):floor(y_max), floor(x_min):floor(x_max), :);
end

%% plot voxel
function [out] = plot_voxel(voxel, voxel_scale)
[shape_vis]=VoxelPlotter(voxel,1,1);
hold on; view(3); axis equal;scatter3(0,0,0);
axis([0,voxel_scale,0,voxel_scale,0,voxel_scale]);
out = true;
end

%% plot voxel twice
function [out] = plot_voxel_twice(voxel, voxel_scale)
subplot(1,3,1);title('voxel gt');
plot_voxel(voxel, voxel_scale);
subplot(1,3,2);title('align');
plot_voxel(voxel, voxel_scale);
h3 = subplot(1,3,3);cla(h3);title('prim');
out = true;
end

%% plot image and bbox
function [prim_num_gt] = plot_image_and_bbox(img_name, img_dir, img_vox_id)
global cls
img_id = str2num(img_name(1:4));
index = find(img_vox_id.img_idxs == img_id);
prim_num_gt = img_vox_id.voxel_idxs(index);
if strcmp(cls, '3dprnnchair') || strcmp(cls, '3dprnntable') || strcmp(cls, '3dprnnnight_stand')
    img = load([img_dir img_name(1:4) '.mat']);
    img = img.depth;
else
    %img = imread([img_dir img_name]);
    img = imread([img_dir img_name(1:4) '.png']);
end
imshow(img);
img_name
end

%% plot image and match voxel id
function [prim_num_gt] = plot_image_and_match_id(img_name, img_dir, img_vox_id)
global cls
img_id = str2num(img_name(1:4));
index = find(img_vox_id.img_idxs == img_id);
prim_num_gt = img_vox_id.voxel_idxs(index);
% if compute iou for tul, no loading or drawing below
if strcmp(cls, '3dprnnchair') || strcmp(cls, '3dprnntable') || strcmp(cls, '3dprnnnight_stand')
    img = load([img_dir img_name(1:4) '.mat']);
    img = img.depth;
else
    %img = imread([img_dir img_name]);
    img = imread([img_dir img_name(1:4) '.png']);
end
subplot(2,3,3);
imshow(img);
title(img_name);
end

%% plot a prim twice for prediction
function [out] = plot_a_prim_twice_pred(prim_r, color, sem_id, voxel_scale)
vertices = prim20_to_vertices(prim_r, voxel_scale);
subplot(2,3,2);
plot_a_prim_align(vertices, color, sem_id, voxel_scale);
subplot(2,3,5);title('gt')
plot_a_prim_alone(vertices, color, sem_id, voxel_scale);
out = true;
end

%% plot a prim twice for editting
function [out] = plot_a_prim_twice_edit(prim_r, color, sem_id, voxel_scale)
vertices = prim20_to_vertices(prim_r, voxel_scale);
subplot(1,3,2);
plot_a_prim_align(vertices, color, sem_id, voxel_scale);
subplot(1,3,3);
plot_a_prim_alone(vertices, color, sem_id, voxel_scale);
out = true;
end

%% plot a prim twice for voxelization
function [out] = plot_a_prim_twice_voxelize(prim_r, color, sem_id, voxel_scale, num_win)
vertices = prim20_to_vertices(prim_r, voxel_scale);
subplot(1,num_win,1);
plot_a_prim_align(vertices, color, sem_id, voxel_scale);
subplot(1,num_win,2);
plot_a_prim_align(vertices, color, sem_id, voxel_scale);
out = true;
end

%% plot a prim to align with voxel
function [out] = plot_a_prim_align(vertices, color, sem_id, voxel_scale)
faces = [1,2,3,4;5,6,7,8;1,2,6,5;3,4,8,7;1,4,8,5;2,3,7,6];
patch('Faces',faces,'Vertices',vertices,'FaceColor',color{sem_id},'FaceAlpha',0.3);
view(3); axis equal; hold on;
axis([0,voxel_scale,0,voxel_scale,0,voxel_scale]);
axis off;
%set(gca,'xtick',0:0:0);
%set(gca,'ytick',0:0:0);
%set(gca,'ztick',0:0:0);
out = true;
end

%% plot a prim alone to show
function [out] = plot_a_prim_alone(vertices, color, sem_id, voxel_scale)
faces = [1,2,3,4;5,6,7,8;1,2,6,5;3,4,8,7;1,4,8,5;2,3,7,6];
light('Position',[-1 -1 0],'Style','local');
patch('Faces',faces,'Vertices',vertices,'FaceColor',color{sem_id},'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.6, 'DiffuseStrength', 0.8, 'FaceAlpha',0.8);
view(3); axis equal; hold on;
axis([0,voxel_scale,0,voxel_scale,0,voxel_scale])
axis off;
%figure(2)
%light('Position',[5 100 15],'Style','local')
        %patch('Faces',faces,'Vertices',vertices,'FaceColor',color{tol_cnt},'FaceAlpha',0.8);
%        patch('Faces',faces,'Vertices',vertices,'FaceColor',[0.75 0.75 0.75],'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.8, 'DiffuseStrength', 0.3, 'FaceAlpha',1);
%        view(3); axis equal; hold on;
        %figure(1)
out = true;
end

%% compute vertices from a prim (1,20)
function [prim_pt] = prim20_to_vertices(prim_r, voxel_scale)
[prim_pt, prim_pt_mean] = prim_to_prim_pt(prim_r(11:16));

Rv = prim_r(:,17:19);
vx = getVX(Rv);% rotation
theta = prim_r(:,20);
Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
prim_pt = bsxfun(@minus, prim_pt, prim_pt_mean);
prim_pt = prim_pt*Rrot;
prim_pt = bsxfun(@plus, prim_pt, prim_pt_mean);
prim_pt = bsxfun(@min,prim_pt,voxel_scale);
end

%% compute vertices from a prim (1,9)
function [prim_pt] = prim9_to_vertices(prim_r, voxel_scale, sym)
global demo_rot
[prim_pt, prim_pt_mean] = prim_to_prim_pt(prim_r);

Rv = prim_r(:,7:9);
[~, Rv_i] = max(abs(Rv));
theta = Rv(Rv_i);
if Rv_i ~= 1 && sym == 1
    theta = -theta;
end
%[Rv_y, Rv_i] = max(sym_r);
Rv = zeros(1,3); Rv(Rv_i) = 1;
%theta = prim_rot(Rv_i);

vx = getVX(Rv);% rotation
Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;

prim_pt = bsxfun(@minus, prim_pt, prim_pt_mean);
prim_pt = prim_pt*Rrot;
prim_pt = bsxfun(@plus, prim_pt, prim_pt_mean);
%prim_pt = bsxfun(@min,prim_pt,voxel_scale);

if demo_rot >= 0
    Rv = zeros(1,3); Rv(3) = 1;
    vx = getVX(Rv);% rotation
    demo_Rrot = cos(demo_rot)*eye(3) + sin(demo_rot)*vx + (1-cos(demo_rot))*Rv'*Rv;
    prim_pt = bsxfun(@minus, prim_pt, [16,16,16]);
    prim_pt = prim_pt * demo_Rrot;
    prim_pt = bsxfun(@plus, prim_pt, [16,16,16]);
end
end

%% prim to prim_pt
function [prim_pt, prim_pt_mean] = prim_to_prim_pt(prim_r)
prim_pt_x = [0 prim_r(1) prim_r(1) 0 0 prim_r(1) prim_r(1) 0];
prim_pt_y = [0 0 prim_r(2) prim_r(2) 0 0 prim_r(2) prim_r(2)];
prim_pt_z = [0 0 0 0 prim_r(3) prim_r(3) prim_r(3) prim_r(3)];
prim_pt = [prim_pt_x' prim_pt_y' prim_pt_z'];
prim_pt = bsxfun(@plus, prim_pt, prim_r(:,4:6));
prim_pt_mean = mean(prim_pt);
end

%% save images
function [out] = save_img(i, save_visual, I, cls, exp, iou)
global save_pred_dir
if save_visual
%         subplot(2,2,2);cla;reset(gca);
%         subplot(2,3,3);
%         imagesc(reshape(dp_ts.depth_tile(i,:,:),64,64)); axis equal; axis([0,64,0,64]);
%         subplot(2,2,4);cla;
    
%         iou_i = iou.iou_chair(i);
%         inters = [0.2, 0.4, 0.6, 0.8];
%         inters = [0.1, 0.2, 0.3, 0.4];
%         iou_i = iou.dist(i);
%         if iou_i < inters(1)
%             dir_i = '1';
%         elseif iou_i < inters(2)
%             dir_i = '2';
%         elseif iou_i < inters(3)
%             dir_i = '3';
%         elseif iou_i < inters(4)
%             dir_i = '4';
%         else
%             dir_i = '5';
%         end
%         saveas(I, sprintf(['/Users/heqian/Research/projects/image_results/' cls '_pred_' exp '/' dir_i '/%d.jpg'], i));
        saveas(I, sprintf([save_pred_dir '/image_results/tul/' cls '/%d.jpg'], i));
%        saveas(I, sprintf([save_pred_dir '/image_results/' cls '_pred_' exp '/%d.jpg'], i));
%         saveas(I, sprintf([save_pred_dir '\\image_results\\' cls '_pred_' exp '\\%d.jpg'], i));
end
out = 1;
end

%% draw result prims demo refinement
function [out] = draw_res_demo_refine(i, inter_i, res1, res2, color, voxel_scale, sem_flag, full_model)
global n_sem len_adjust
hold off;cla;
tol_cnt = 1;

correspondence = [1,3,8,2,7,6,4,5];

res_x1 = res1.x;
res_cls1 = res1.cls;
res_prim1 = res_x1((i-1)*4+1:(i-1)*4+2, :);
res_rot1 = res_x1((i-1)*4+3, :);
stop_idx = find(res_prim1(1,:) == 0,1,'first');

res_x2 = res2.x;
res_cls2 = res2.cls;
res_prim2 = res_x2((i-1)*4+1:(i-1)*4+2, :);
res_rot2 = res_x2((i-1)*4+3, :);

add=1;
start = add*3+1-3;
tol_cnt = add;
for res_row = start:3:stop_idx-3
    prim_rot1 = res_rot1(res_row:res_row+2);
    prim_r1 = [res_prim1(1, res_row:res_row+2) res_prim1(2, res_row:res_row+2) prim_rot1];
    res_row2 = correspondence(round((res_row-1)/3+1));
    if res_row2 == 8
        prim_r2 = [0,0,0,prim_r1(4:9)];
    else
        res_row2 = round((res_row2-1)*3+1);
        prim_rot2 = res_rot2(res_row2:res_row2+2);
        prim_r2 = [res_prim2(1, res_row2:res_row2+2) res_prim2(2, res_row2:res_row2+2) prim_rot2];
    end
    prim_r = prim_r1*(1-inter_i) + prim_r2*inter_i;
    
    if sem_flag
        sem_id = res_cls1(i, res_row) + 1;
        if sem_id > n_sem && len_adjust
            continue
        end
    else
        sem_id = res_cls1(i, res_row) + 1;
        if sem_id > 1 && len_adjust
            continue
        end
        sem_id = tol_cnt;
    end
    %sem_id = add;
    vertices = prim9_to_vertices(prim_r, voxel_scale, 0);
    plot_a_prim_alone(vertices, color, sem_id, voxel_scale);
    
    %if  15 - prim_r(4)- prim_r(1)/2> 3 %sum(sym_r > 0.5)>1.5
    if prim_r(4)+prim_r(1) < voxel_scale/2 && ~full_model
        prim_r(4) = voxel_scale - prim_r(4)-prim_r(1);%/2;%%%%%heqian
        vertices = prim9_to_vertices(prim_r, voxel_scale, 1);
        plot_a_prim_alone(vertices, color, sem_id, voxel_scale);
    end
    tol_cnt = tol_cnt + 1;
end
out = stop_idx - 1;
end

%% draw result prims demo
function [out] = draw_res_demo(i, res, color, voxel_scale, iou, sem_flag, full_model)
global n_sem len_adjust
hold off;cla;
tol_cnt = 1;

% i = i - 1;
% repeat = 10;
res_x = res.x;
res_cls = res.cls;
res_prim = res_x((i-1)*4+1:(i-1)*4+2, :);
res_rot = res_x((i-1)*4+3, :);
res_sym = res_x(i*4, :);
stop_idx = find(res_prim(1,:) == 0,1,'first');

add=1;
start = add*3+1-3;
tol_cnt = add;
for res_row = start:3:stop_idx-3%-3%%%nms
    prim_rot = res_rot(res_row:res_row+2);
    prim_r = [res_prim(1, res_row:res_row+2) res_prim(2, res_row:res_row+2) prim_rot];
    sym_r = res_sym(res_row:res_row+2); 
    if sem_flag
        sem_id = res_cls(i, res_row) + 1;
        if sem_id > n_sem && len_adjust
            continue
        end
    else
        sem_id = res_cls(i, res_row) + 1;
        if sem_id > 1 && len_adjust
            continue
        end
        sem_id = tol_cnt;
    end
    %sem_id = add;
    vertices = prim9_to_vertices(prim_r, voxel_scale, 0);
    plot_a_prim_alone(vertices, color, sem_id, voxel_scale);
    
    %if  15 - prim_r(4)- prim_r(1)/2> 3 %sum(sym_r > 0.5)>1.5
    if prim_r(4)+prim_r(1) < voxel_scale/2 && ~full_model
        prim_r(4) = voxel_scale - prim_r(4)-prim_r(1);%/2;%%%%%heqian
        vertices = prim9_to_vertices(prim_r, voxel_scale, 1);
        plot_a_prim_alone(vertices, color, sem_id, voxel_scale);
    end
    tol_cnt = tol_cnt + 1;
end
out = stop_idx - 1;
iou
end

%% draw result prims
function [out] = draw_res(i, res, color, voxel_scale, win_id, iou, sem_flag, full_model)
% subplot(2,3,win_id); title(iou.iou_chair(i)); % title('generation')% pred
global n_sem len_adjust nms
subplot(2,3,win_id); title(iou.dist(i)); % title('generation')% pred
hold off;
if win_id == 3
    cla;
end
tol_cnt = 1;

% i = i - 1;
% repeat = 10;
res_x = res.x;
res_cls = res.cls;
res_prim = res_x((i-1)*4+1:(i-1)*4+2, :);
res_rot = res_x((i-1)*4+3, :);
res_sym = res_x(i*4, :);
stop_idx = find(res_prim(1,:) == 0,1,'first');

add=1;
start = add*3+1-3;
tol_cnt = add;
for res_row = start:3:stop_idx-3
    prim_rot = res_rot(res_row:res_row+2);
    prim_r = [res_prim(1, res_row:res_row+2) res_prim(2, res_row:res_row+2) prim_rot];
    sym_r = res_sym(res_row:res_row+2); 
    if sem_flag
        sem_id = res_cls(i, res_row) + 1;
        if sem_id > n_sem && len_adjust
            continue
        end
    else
        sem_id = res_cls(i, res_row) + 1;
        if sem_id > 1 && len_adjust
            continue
        end
        sem_id = tol_cnt;
    end
    if sem_id > size(color, 2)
        sem_id = sem_id - size(color, 2);
    end
    if sum(iou.nms_removed(i, :) + 1 == tol_cnt) > 0 && nms
        sem_id = 13;
    end
    %sem_id = add;
    vertices = prim9_to_vertices(prim_r, voxel_scale, 0);
    if win_id == 1
        plot_a_prim_align(vertices, color, sem_id, voxel_scale);
    else
        plot_a_prim_alone(vertices, color, sem_id, voxel_scale);
    end
    %if  15 - prim_r(4)- prim_r(1)/2> 3 %sum(sym_r > 0.5)>1.5
    if prim_r(4)+prim_r(1) < voxel_scale/2 && ~full_model
        prim_r(4) = voxel_scale - prim_r(4)-prim_r(1);%/2;%%%%%heqian
        vertices = prim9_to_vertices(prim_r, voxel_scale, 1);
        if win_id == 1
            plot_a_prim_align(vertices, color, sem_id, voxel_scale);
        else
            plot_a_prim_alone(vertices, color, sem_id, voxel_scale);
        end
    end
    tol_cnt = tol_cnt + 1;
end
out = stop_idx - 1;
end


%% prim length plot
function [out] = plot_prim_num(cls, exp)
length_val = load(['/Users/heqian/Research/projects/image_results/' cls '_pred_' exp '/length_val.mat']);
length_val = length_val.length_gt;
length_train = load(['/Users/heqian/Research/projects/image_results/' cls '_pred_' exp '/length_train.mat']);
length_train = length_train.length_gt;
num_val = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
num_train = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
for i = 1:size(length_val, 1)
    num_val(length_val(i, 1)) = num_val(length_val(i, 1)) + 1;
end
for i = 1:size(length_train, 1)
    num_train(length_train(i, 1)) = num_train(length_train(i, 1)) + 1;
end
x = 1:1:20;
% plot(x, num_val, x, num_train);
plot(x, num_val/size(length_val, 1), x, num_train/size(length_train, 1));
out = 1;
end

%% prim length acc
% [error, error_all] = compute_acc(cls, exp);
function [error, error_all] = compute_acc(cls, exp)
% length_val = load(['/Users/heqian/Research/projects/image_results/' cls '_pred_' exp '/length_val.mat']);
% length_val = length_val.length_gt;
length = load(['/Users/heqian/Research/projects/image_results/' cls '_pred_' exp '/length.mat']);
length = length.length;
error = 0;
miss = 0;
more = 0;
less = 0;
sum = 0;
valid_count = 0;
for i = 1:size(length, 1)
    % length(i, 1):pred; length(i, 2):gt;
%     if length_val(i, 1) ~= 4
    miss = miss + abs(length(i, 1) - length(i, 2));
    if length(i, 1) - length(i, 2) > 0
        more = more + 1;
    elseif length(i, 1) - length(i, 2) < 0
        less = less + 1;
    end
    sum = sum + length(i, 2);
    error_i = abs(length(i, 1) - length(i, 2)) / length(i, 2);
    error = error + error_i;
    valid_count = valid_count + 1;
%     end
end
error = error / size(length, 1);
error_all = miss / sum;
more_ratio = more / size(length, 1);
less_ratio = less / size(length, 1);
equal_ratio = 1 - more_ratio - less_ratio;
equal = size(length, 1) - more - less;
end