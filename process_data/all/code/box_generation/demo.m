% demo to run cuboid fitting
% "prim_gt" is the predicted primitive, with the format "[mesh number, shape, transition, rotation]"
% If fitting fails due to random optimization, prim_gt is all zero
%
% Currently one primitive fitting step. Will update sooner.

addpath(genpath('./minFunc_2012/'));

cls = 'chair';
root = '/Users/heqian/Research/projects/primitive-based_3d/process_data/all';
out_root = [root '/output/' cls];
out_mat_dir = [out_root '/mat'];
mkdir(out_mat_dir);
vox_data = load([out_root '/downsampled_voxels_' cls '.mat']);
vox_data = vox_data.voxTile;
voxel_scale = 32;%30;
sample_grid = 7;
prim_end_limit = 20;
iter_limit = 2;
prim_num_limit = 12;
refine_limit = 0;%3;
thresh = 0.3;
%optimization
options = [];
options.display = 'none';
options.MaxIter = 100;
options.Method = 'lbfgs';
options.LS_init = 2;
options.useMex = 0;%heqian

tic
primset = {};
primset_refine = {};
primset_savename = [out_mat_dir '/Myprimset.mat'];
primset_savename_refine = [out_mat_dir '/Myprimset_refine.mat'];
for gt_num = 1:size(vox_data,1)     % loop through all objects within the dataset
    %tic
    gt_num
    prim_save = [];
    vox_savename = [out_mat_dir '/Myvox' num2str(gt_num) '.mat'];
    prim_savename = [out_mat_dir '/Myprim' num2str(gt_num) '.mat'];
    prim_save_refine = [];
    vox_savename_refine = [out_mat_dir '/Myvox_refine' num2str(gt_num) '.mat'];
    prim_savename_refine = [out_mat_dir '/Myprim_refine' num2str(gt_num) '.mat'];
    gt_savename = 'Mygt.mat';
    
    gt_voxel = reshape(vox_data(gt_num,:,:,:), voxel_scale, voxel_scale, voxel_scale);
    %gt_voxel = permute(gt_voxel, [3,2,1]);
    [gt] = compute_save_gt_pt(gt_savename, voxel_scale, gt_voxel);
    vox_remained = gt_voxel;
    vox_remained_refine = gt_voxel;
    new_vox = [];
    new_vox_refine = [];
    
    %%heqian
    prim_count = 0;
    prim_count_refine = 0;
    refine_switch = 0;
    refine_count = 0;
    gt_pt_num = size(gt.gt_pt, 1);
    beta = 0.03 * gt_pt_num;
    while size(gt.gt_pt, 1) > beta && prim_count < prim_num_limit     % loop through all primitives within one object
        iter = 1;
        prim_end_flag = 0;
        prim_iter = [];
        while iter <= iter_limit     % loop for one primitive within one object
            [shape, trans, Rv, theta, prim_gt, affv] = initialize_st(gt.gt_pt, voxel_scale, gt_num, refine_switch, refine_count, prim_save, prim_count);%%%refine
            stop_thres = 1e-1;
            cnt = 1;
            while 1     % loop for optimization of st and r for one primitive within one object
                x = minFunc(@sampleEijOpt,[affv';Rv],options);
                if norm(x(1:6) - affv')  < stop_thres % converge
                    break
                end
                affv = x(1:6)';
                % quit wrong result
                shape = x(1:3); trans = x(4:6)';
                if sum(shape<0) > 0 || shape(1) > voxel_scale || shape(2) > voxel_scale || shape(3) > voxel_scale %27000
                    if sum(shape<0) > 0
                        fprintf('shape < 0\n');
                    else
                        fprintf('shape(:) > %d\n',voxel_scale);
                    end
                    prim_end_flag = prim_end_flag + 1;
                    break
                end
                [sample_pt] = sample_primitive(shape, sample_grid);
                % refinement on rotation
                [Rv] = refine_rotation(sample_pt, trans, shape, Rv, theta, gt);
                if cnt > 4
                    break
                end
                cnt = cnt +1;
            end
            renew_gt = 0;
            if sum(shape<0) > 0 || shape(1) > voxel_scale || shape(2) > voxel_scale || shape(3) > voxel_scale %27000
                prim_gt(1,11:end) = zeros(1,9);
            else
                x = minFunc(@sampleEijOpt2,[x(1:6);Rv],options);
                prim_gt(1,11:end) = [ x(1:6)' Rv'];
                renew_gt = 1;
            end
            if renew_gt
                [loss, ~] = sampleEijOpt2(x);
                prim_iter(iter, :) = [prim_gt, loss];
                iter = iter + 1;
            end
            if prim_end_flag >= prim_end_limit
                break
            end
        end
        if prim_end_flag >= prim_end_limit && iter <= 1
            fprintf('prim_end_flag >= 20 && iter <= 1\n');
            break
        end
        prim_iter
        [~, minidx] = min(prim_iter(:, 20));
        prim_gt = prim_iter(minidx, 1:19);
        prim_gt = crop_prim(prim_gt, voxel_scale);
        %%heqian
        %%sample_core_Eij_sum2(sample_pt, trans, shape, scale, Rv, theta, gt)
        sym_count = 0;
        while sym_count < 2%renew_gt     % loop through a pair of symmetric primitives within one object
            fprintf('refine_switch %d\n', refine_switch);
            if refine_switch
                refine_count = refine_count - 1;
                [gt, vox_deleted, vox_remained_refine, Rv, theta] = save_primcheck(gt, prim_gt, sample_grid, thresh, vox_remained_refine, voxel_scale);
                fprintf('gt.gt_pt %d\n', size(gt.gt_pt, 1));
                prim_count_refine = prim_count_refine + 1;
                new_vox_refine.deleted(prim_count_refine, :, :, :) = vox_deleted;
                new_vox_refine.remained(prim_count_refine, :, :, :) = vox_remained_refine;
                save(vox_savename_refine, 'new_vox_refine');
                [prim_save_refine, prim_gt, sym_count, break_flag] = save_prim(prim_save_refine, prim_gt, prim_count_refine, sym_count, voxel_scale, Rv, theta);
                save(prim_savename_refine, 'prim_save_refine');
            else
                refine_count = refine_count + 1;
                [gt, vox_deleted, vox_remained, Rv, theta] = save_primcheck(gt, prim_gt, sample_grid, thresh, vox_remained, voxel_scale);
                fprintf('gt.gt_pt %d\n', size(gt.gt_pt, 1));
                prim_count = prim_count + 1;
                new_vox.deleted(prim_count, :, :, :) = vox_deleted;
                new_vox.remained(prim_count, :, :, :) = vox_remained;
                save(vox_savename, 'new_vox');
                [prim_save, prim_gt, sym_count, break_flag] = save_prim(prim_save, prim_gt, prim_count, sym_count, voxel_scale, Rv, theta);
                save(prim_savename, 'prim_save');
            end
            if refine_count >= refine_limit && sym_count ~= 1 && refine_limit > 0
                refine_switch = 1;
            elseif refine_count == 0
                refine_switch = 0;
                vox_remained = vox_remained_refine;
            end
            if refine_switch
                [gt] = add_prim_to_gt(gt, squeeze(new_vox.deleted(prim_count - refine_count + 1, :, :, :)));
            end
            save(gt_savename,'gt');
            if break_flag
                break
            end
        end
    end

    if size(prim_save_refine, 1) > 0
        primset_tmp = save_primset(prim_save_refine, gt_num);
        primset_refine{gt_num, 1} = primset_tmp{1, 1};
        save(primset_savename_refine, 'primset_refine');
    end
    if size(prim_save, 1) > 0
        primset_tmp = save_primset(prim_save, gt_num);
        primset{gt_num, 1} = primset_tmp{1, 1};
        save(primset_savename, 'primset');
    end
    toc
end


function [prim_save, prim_gt, sym_count, break_flag] = save_prim(prim_save, prim_gt, prim_count, sym_count, voxel_scale, Rv, theta)
break_flag = 0;
if prim_gt(1, 11)+prim_gt(1, 14) < voxel_scale/2
    side = 0;
    prim_save(prim_count, :) = [prim_gt(1, 1:16), Rv, theta, side];
%     save(prim_savename, 'prim_save');
    prim_gt(1, 14) = voxel_scale - prim_gt(1, 14) - prim_gt(1, 11);
    [~, Rv_t] = max(abs(Rv));
    if Rv_t ~=1
        prim_gt(1, 16+Rv_t) = -prim_gt(1, 16+Rv_t);
    end
    sym_count = sym_count + 1;
elseif prim_gt(1, 14) > voxel_scale/2
    side = 1;
    prim_save(prim_count, :) = [prim_gt(1, 1:16), Rv, theta, side];
%     save(prim_savename, 'prim_save');
    prim_gt(1, 14) = voxel_scale - prim_gt(1, 14) - prim_gt(1, 11);
    [~, Rv_t] = max(abs(Rv));
    if Rv_t ~=1
        prim_gt(1, 16+Rv_t) = -prim_gt(1, 16+Rv_t);
    end
    sym_count = sym_count + 1;
else
    prim_save(prim_count, :) = [prim_gt(1, 1:16), Rv, theta, -1];
%     save(prim_savename, 'prim_save');
    break_flag = 1;
end
end


function [prim_gt] = crop_prim(prim_gt, voxel_scale)
for ax = 1:3
    if prim_gt(1, 13+ax) < 0
        prim_gt(1, 10+ax) = prim_gt(1, 10+ax) + prim_gt(1, 13+ax);
        prim_gt(1, 13+ax) = 0;
    end
    if prim_gt(1, 10+ax) + prim_gt(1, 13+ax) > voxel_scale
        prim_gt(1, 10+ax) = voxel_scale - prim_gt(1, 13+ax);
    end
end
end

function [gt] = add_prim_to_gt(gt, vox_deleted)
count = 0;
new_gt_pt = gt.gt_pt;
new_gt_pt_inv = gt.gt_pt_inv;
for gt_pt_inv_i = 1:size(gt.gt_pt_inv, 1)
    point = gt.gt_pt_inv(gt_pt_inv_i, :);
    if vox_deleted(point(1),point(2),point(3))
        new_gt_pt = [new_gt_pt; point];
        new_gt_pt_inv(gt_pt_inv_i-count, :) = [];
        count = count + 1;
    end
end
gt.gt_pt = new_gt_pt;
gt.gt_pt_inv = new_gt_pt_inv;
fprintf('prim_vox_count: %d   gt.gt_pt %d\n', count, size(gt.gt_pt, 1));
% for i = 1:voxel_scale
%     for j = 1:voxel_scale
%         for k = 1:voxel_scale
%             if vox_deleted(i,j,k)
%                 count = count + 1;
%                 gt.gt_pt(start+count, :) = [i,j,k];
%                 %gt.gt_pt_inv
%             end
%         end
%     end
% end
end

function [shape, trans, Rv, theta, cen_mean] = decompose_prim(prim_gt, sample_grid)
scale = [1,1,1];
shape = prim_gt(1,11:13)';
trans = prim_gt(1,14:16);
Rv = prim_gt(1,17:19)';
[Rv, theta] = rv_to_rv_theta(Rv);
% affine transformation
vx = getVX(Rv);% rotation
Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
[sample_pt] = sample_primitive(shape, sample_grid);
[cen_mean, ~] = get_mean_and_pt(sample_pt, trans, scale, Rrot);
end


function [gt, vox_deleted, vox_remained, Rv, theta] = save_primcheck(gt, prim_gt, sample_grid, thresh, vox_remained, voxel_scale)
[shape, trans, Rv, theta, cen_mean] = decompose_prim(prim_gt, sample_grid);
gt_pt = gt.gt_pt;
gt_pt_inv = gt.gt_pt_inv;
new_gt_pt = gt_pt;
new_gt_pt_inv = gt_pt_inv;
% min_sample_pt = min(sample_pt_dst);
% max_sample_pt = max(sample_pt_dst)+1;
delete_count = 0;
gt_pt_deleted = [];
for gt_i = 1:size(gt_pt, 1)
    [in_box, ~] = check_voxel_in_box(gt_pt(gt_i, :), shape, trans, Rv, theta, cen_mean, thresh);
    if in_box
        new_gt_pt(gt_i-delete_count, :) = [];
        delete_count = delete_count + 1;
        gt_pt_deleted(delete_count, :) = gt_pt(gt_i, :);
        new_gt_pt_inv = [new_gt_pt_inv; gt_pt(gt_i, :)];
    end
end
gt_pt = new_gt_pt;
gt_pt_inv = new_gt_pt_inv;
gt.gt_pt = gt_pt;
gt.gt_pt_inv = gt_pt_inv;
vox_deleted = zeros(voxel_scale, voxel_scale, voxel_scale);
for vox_i = 1:size(gt_pt_deleted,1)
    vox_deleted(gt_pt_deleted(vox_i,1), gt_pt_deleted(vox_i,2), gt_pt_deleted(vox_i,3)) = 1;
    vox_remained(gt_pt_deleted(vox_i,1), gt_pt_deleted(vox_i,2), gt_pt_deleted(vox_i,3)) = 0;
end
end


function [Rv, theta] = rv_to_rv_theta(Rv)
if Rv(1) == 0 && Rv(2) == 0 && Rv(3) == 0
    theta = 0;
    Rv = zeros(1,3);
else
    [~, Rv_i] = max(abs(Rv));
    theta = Rv(Rv_i);
    Rv = zeros(1,3); Rv(Rv_i) = 1;
end
end


function [Rv] = refine_rotation(sample_pt, trans, shape, Rv, theta, gt)
Rv = Rv/(theta+eps);
[loss, sample_pt_dst] = sample_core_Eij_sum2(sample_pt, trans, shape, [1,1,1], Rv', theta, gt);
loss_max = loss;
theta_max = theta;
Rv_max = Rv';
Rv_r = [1,0,0];
for i = -0.54:0.06:0.54
    if norm(Rv'-Rv_r) < 0.5
        theta_r = theta + i;
    else
        theta_r = i;
    end
    [loss_r, sample_pt_dst_r] = sample_core_Eij_sum2(sample_pt, trans, shape, [1,1,1], Rv_r, theta_r, gt);
    if loss_r > loss_max
        loss_max = loss_r;
        theta_max = theta_r;
        Rv_max = Rv_r;
        sample_pt_dst = sample_pt_dst_r;
    end
end
Rv_r = [0,1,0];
for i = -0.54:0.18:0.54
    if norm(Rv'-Rv_r) < 0.5
        theta_r = theta + i;
    else
        theta_r = i;
    end
    [loss_r, sample_pt_dst_r] = sample_core_Eij_sum2(sample_pt, trans, shape, [1,1,1], Rv_r, theta_r, gt);
    if loss_r > loss_max
        loss_max = loss_r;
        theta_max = theta_r;
        Rv_max = Rv_r;
        sample_pt_dst = sample_pt_dst_r;
    end
end
Rv_r = [0,0,1];
for i = -0.54:0.18:0.54
    if norm(Rv'-Rv_r) < 0.5
        theta_r = theta + i;
    else
        theta_r = i;
    end
    [loss_r, sample_pt_dst_r] = sample_core_Eij_sum2(sample_pt, trans, shape, [1,1,1], Rv_r, theta_r, gt);
    if loss_r > loss_max
        loss_max = loss_r;
        theta_max = theta_r;
        Rv_max = Rv_r;
        sample_pt_dst = sample_pt_dst_r;
    end
end
Rv = Rv_max*theta_max;
Rv = Rv';
%theta = theta_max;
end


function [cen_mean, sample_pt_dst] = get_mean_and_pt(sample_pt, trans, scale, Rrot)
sample_pt_dst = bsxfun(@plus, sample_pt, trans);
cen_mean = mean(sample_pt_dst);
sample_pt_dst = bsxfun(@minus, sample_pt_dst, cen_mean);
sample_pt_dst = bsxfun(@times, sample_pt_dst, scale);
sample_pt_dst = sample_pt_dst * Rrot;
sample_pt_dst = bsxfun(@plus, sample_pt_dst, cen_mean);
end


function [sample_pt] = sample_primitive(shape, sample_grid)
sample_dist_x = shape(1)/sample_grid;% sampling 
sample_x = 0:sample_dist_x:shape(1);
sample_dist_y = shape(2)/sample_grid;
sample_y = 0:sample_dist_y:shape(2);
sample_dist_z = shape(3)/sample_grid;
sample_z = 0:sample_dist_z:shape(3);
[sample_X,sample_Y,sample_Z] = meshgrid(sample_x,sample_y,sample_z);
sample_pt = [sample_X(:), sample_Y(:), sample_Z(:)];
end


function [shape, trans, Rv, theta, prim_gt, affv] = initialize_st(gt_pt, voxel_scale, gt_num, refine_switch, refine_count, prim_save, prim_count)
if refine_switch
    prim_gt = prim_save(prim_count - refine_count + 1, :);
    shape = prim_gt(1, 11:13);
    trans = prim_gt(1, 14:16);
    Rv = prim_gt(1, 17:19);
    theta = prim_gt(1, 20);
    Rv = Rv * theta;
    prim_gt = [prim_gt(1, 1:16) Rv];
    Rv = Rv';
    affv = prim_gt(1,11:16);
else
    rng('shuffle');
    trans = randperm(size(gt_pt,1)); % random initialization
    trans = gt_pt(trans(1),:);
    shape = rand(1,3)*voxel_scale/3+3;
    bound = voxel_scale - 5;
    if shape(3) + trans(3) > bound
        trans(3) = trans(3) - (shape(3) + trans(3)-bound);
    end
    if shape(2) + trans(2) > bound
        trans(2) = trans(2) - (shape(2) + trans(2)-bound);
    end
    if shape(1) + trans(1) > bound
        trans(1) = trans(1) - (shape(1) + trans(1)-bound);
    end
    prim_gt = zeros(1, 19);
    prim_gt(1,1:10) = [gt_num, shape, trans, 0,0,0];
    Rv = [0;0;0];
    theta = 0;
    affv = prim_gt(1,2:7);
end
end


function [gt] = compute_save_gt_pt(gt_savename, voxel_scale, gt_voxel)
x_gt = []; y_gt = []; z_gt = [];
x_gt_inv = []; y_gt_inv = []; z_gt_inv = [];
for i = 1:voxel_scale
    % positive
    [x_gt_tmp, y_gt_tmp] = find(gt_voxel(:,:,i)); % define Qj
    x_gt = [x_gt;x_gt_tmp];
    y_gt = [y_gt;y_gt_tmp];
    z_gt = [z_gt;i*ones(size(x_gt_tmp))];
    % negative
    [x_gt_tmp, y_gt_tmp] = find(~gt_voxel(:,:,i)); % define Qj
    x_gt_inv = [x_gt_inv;x_gt_tmp];
    y_gt_inv = [y_gt_inv;y_gt_tmp];
    z_gt_inv = [z_gt_inv;i*ones(size(x_gt_tmp))];
end
% positive 
gt_pt = [x_gt, y_gt, z_gt];
gt.gt_pt = gt_pt;
% negative
gt_pt_inv = [x_gt_inv, y_gt_inv, z_gt_inv];
gt.gt_pt_inv = gt_pt_inv;
save(gt_savename,'gt');
fprintf('gt.gt_pt %d\n', size(gt.gt_pt, 1));
end

function [primset] = save_primset(prim_save, gt_num)
primset = {};
prim_cheight = prim_save(:, 13)/2.0 + prim_save(:, 16);
[heights, height_idx] = sort(prim_cheight, 'descend');
ori_id = 1;
id = 1;
while id <= size(height_idx, 1)
    idx = height_idx(id);
    if prim_save(idx, 21) == -1
        primset{1, 1}.ori(ori_id, :) = prim_save(idx, 1:20);
        primset{1, 1}.cen(ori_id, :) = primset{1, 1}.ori(ori_id, 11:13)/2.0 + primset{1, 1}.ori(ori_id, 14:16);
        primset{1, 1}.side(ori_id) = 0;
        primset{1, 1}.sym(ori_id, :) = zeros(1, 20);
        id = id + 1;
        ori_id = ori_id + 1;
    elseif prim_save(idx, 21) == 0
        primset{1, 1}.ori(ori_id, :) = prim_save(idx, 1:20);
        primset{1, 1}.cen(ori_id, :) = primset{1, 1}.ori(ori_id, 11:13)/2.0 + primset{1, 1}.ori(ori_id, 14:16);
        primset{1, 1}.side(ori_id) = 0;
        id = id + 1;
        idx_sym = height_idx(id);
        if prim_save(idx_sym, 21) == 1 && heights(id) == heights(id-1)
            primset{1, 1}.sym(ori_id, :) = prim_save(idx_sym, 1:20);
            id = id + 1;
        else
            gt_num
        end
        ori_id = ori_id + 1;
    else
        primset{1, 1}.sym(ori_id, :) = prim_save(idx, 1:20);
        primset{1, 1}.side(ori_id) = 0;
        id = id + 1;
        idx_ori = height_idx(id);
        if prim_save(idx_ori, 21) == 0 && heights(id) == heights(id-1)
            primset{1, 1}.ori(ori_id, :) = prim_save(idx_ori, 1:20);
            primset{1, 1}.cen(ori_id, :) = primset{1, 1}.ori(ori_id, 11:13)/2.0 + primset{1, 1}.ori(ori_id, 14:16);
            id = id + 1;
        else
            gt_num
        end
        ori_id = ori_id + 1;
    end
end
end


function [in_box, point] = check_voxel_in_box(point, shape, trans, Rv, theta, cen_mean, thresh)
in_box = 0;
point = point - 0.5;
point = bsxfun(@plus, point, -cen_mean);
theta = -theta;
vx = getVX(Rv);
Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
point = point * Rrot;
point = bsxfun(@minus, point, -cen_mean);
point = bsxfun(@plus, point, -trans);

% voxel = bsxfun(@plus, voxel, -trans);
% %cen_mean = mean(sample_pt_dst);
% voxel = bsxfun(@minus, voxel, -cen_mean);
% %voxel =  bsxfun(@times, voxel, scale);
% theta = -theta;
% vx = getVX(Rv);
% Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
% voxel = voxel * Rrot;
% voxel = bsxfun(@plus, voxel, -cen_mean);
if (point(1) >= 0 || abs(point(1)) < thresh) && (point(1) <= ceil(shape(1)) || abs(point(1) - shape(1)) < thresh)
    if (point(2) >= 0 || abs(point(2)) < thresh) && (point(2) <= ceil(shape(2)) || abs(point(2) - shape(2)) < thresh)
        if (point(3) >= 0 || abs(point(3)) < thresh) && (point(3) <= ceil(shape(3)) || abs(point(3) - shape(3)) < thresh)
            in_box = 1;
        end
    end
end
point = ceil(point);
% for i = 1:3
%     if voxel(i) - floor(voxel(i)) < 0.5
%         voxel(i) = floor(voxel(i));
%     else
%         voxel(i) = ceil(voxel(i));
%     end
% end


%sample_pt_dst = bsxfun(@plus, sample_pt, trans);
%cen_mean = mean(sample_pt_dst);
%sample_pt_dst = bsxfun(@minus, sample_pt_dst, cen_mean);
%sample_pt_dst = bsxfun(@times, sample_pt_dst, scale);
%sample_pt_dst = sample_pt_dst * Rrot;
%sample_pt_dst = bsxfun(@plus, sample_pt_dst, cen_mean);

end
%%heqian 6.12