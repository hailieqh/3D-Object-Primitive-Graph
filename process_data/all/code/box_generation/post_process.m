cls = 'bed';
local_root = '/Users/heqian/Research/projects/primitive-based_3d';
root = [local_root '/process_data/all'];
output_dir = [root '/output/' cls];
edit_dir = [output_dir '/mat'];
sem_flag = 1;
for i = 3:1:6000
    load([edit_dir '/Myprim' num2str(i) '.mat'], 'prim_save');
    [success] = check_post_prim(prim_save, i);
    count = 0;
    for j = 1:size(prim_save, 1)
        if prim_save(j-count, 10) == 11
            prim_save(j-count,:) = [];
            count = count + 1;
        end
    end
    save([edit_dir '/Myprim' num2str(i) '.mat'], 'prim_save');
    if exist([edit_dir '/Myprimset.mat'], 'file')
        load([edit_dir '/Myprimset.mat'], 'primset');
    end
    primset_tmp = save_primset(prim_save, i, sem_flag);
    check_swivel_chair_rot(primset_tmp);
    primset{i, 1} = primset_tmp{1, 1};
    save([edit_dir '/Myprimset.mat'], 'primset');
end


% divide = 1;
% cls =  'chair';
% primset_train = {};
% primset_val = {};
% if divide
%     train_id = 1;дк
%     val_id = 1;
%     for i = 1:62
%         rng('shuffle');
%         if rand < 0.8
%             primset_train(train_id, :) = primset(i, :);
%             train_id = train_id + 1;
%         else
%             primset_val(val_id, :) = primset(i, :);
%             val_id = val_id + 1;
%         end
%     end
%     primset = primset_train;
%     save(['./' dir '/prim_sort_mn_' cls '_train.mat'],'primset');
%     primset = primset_val;
%     save(['./' dir '/prim_sort_mn_' cls '_val.mat'],'primset');
% end

% for i = 1:10
%     load(['./post_primset/Myprim' num2str(i) '.mat'], 'prim_save');
%     count = 0;
%     for j = 1:size(prim_save, 1)
%         if prim_save(j-count, 10) == 11
%             prim_save(j-count,:) = [];
%             count = count + 1;
%         end
%     end
%     save(['./post_primset/Myprim' num2str(i) '.mat'], 'prim_save');
%     primset_tmp = save_primset(prim_save, i, sem_flag);
%     primset{i, 1} = primset_tmp{1, 1};
%     save('./post_primset/Myprimset.mat', 'primset');
% end

function [success] = check_post_prim(prim_save, i)
for pid = 1:size(prim_save, 1)
    if prim_save(pid,1) ~= i
        fprintf('No. %d object with wrong prim id %d --->\n', i, pid);
    end
    if prim_save(pid, 11) < 0 || prim_save(pid, 11)+prim_save(pid, 14) > 32
        fprintf('No. %d object with wrong prim x %d --->\n', i, pid);
    end
    if prim_save(pid, 12) < 0 || prim_save(pid, 12)+prim_save(pid, 15) > 32
        fprintf('No. %d object with wrong prim y %d --->\n', i, pid);
    end
    if prim_save(pid, 13) < 0 || prim_save(pid, 13)+prim_save(pid, 16) > 32
        fprintf('No. %d object with wrong prim z %d --->\n', i, pid);
    end
    if (pid+1) <= size(prim_save, 1)
        if prim_save(pid, 12) == prim_save(pid+1, 12) && prim_save(pid, 13) == prim_save(pid+1, 13) && prim_save(pid, 15) == prim_save(pid+1, 15)
            if prim_save(pid, 11) ~= prim_save(pid+1, 11)
                fprintf('type A No. %d object with wrong prim sym %d --->\n', i, pid);
            end
            if abs(prim_save(pid, 11) + prim_save(pid, 14) + prim_save(pid+1, 14) - 32) > 0.0001
                fprintf('type B No. %d object with wrong prim sym %d --->\n', i, pid);
            end
            if prim_save(pid, 21) + prim_save(pid+1, 21) ~= 1
                fprintf('type C No. %d object with wrong prim sym %d --->\n', i, pid);
            end
        end
    end
    if prim_save(pid, 17) + prim_save(pid, 18) + prim_save(pid, 19) > 1
        fprintf('No. %d object with wrong prim ax %d --->\n', i, pid);
    end
    if prim_save(pid, 17) + prim_save(pid, 18) + prim_save(pid, 19) == 1 && prim_save(pid, 20) == 0
        fprintf('No. %d object with wrong prim theta %d --->\n', i, pid);
    end
    if prim_save(pid, 17) + prim_save(pid, 18) + prim_save(pid, 19) == 0 && prim_save(pid, 20) ~= 0
        fprintf('No. %d object with wrong prim theta %d --->\n', i, pid);
    end
end
success = 1;
end

function [out] = check_swivel_chair_rot(primset)
prims = primset{1, 1}.ori;
clses = primset{1, 1}.cls;
pre = pi/4;
for i = 1:size(prims, 1)
    if clses(i) == 6
        if prims(i, 20) > pre
            sprintf('error')
        else
            pre = prims(i, 20);
        end
    end
end
end

function [primset] = save_primset(prim_save, gt_num, sem_flag)
primset = {};
prim_cheight = prim_save(:, 13)/2.0 + prim_save(:, 16);
[heights, height_idx] = sort(prim_cheight, 'descend');
[heights, height_idx] = change_idxs_in_rotation_order(heights, height_idx, prim_save, sem_flag);
ori_id = 1;
id = 1;
while id <= size(height_idx, 1)
    idx = height_idx(id);
    if prim_save(idx, 21) == -1
        primset{1, 1}.ori(ori_id, :) = prim_save(idx, 1:20);
        primset{1, 1}.cen(ori_id, :) = primset{1, 1}.ori(ori_id, 11:13)/2.0 + primset{1, 1}.ori(ori_id, 14:16);
        primset{1, 1}.side(ori_id) = 0;
        primset{1, 1}.sym(ori_id, :) = zeros(1, 20);
        if sem_flag
            primset{1, 1}.cls(ori_id) = prim_save(idx, 22);
        end
        id = id + 1;
        ori_id = ori_id + 1;
    elseif prim_save(idx, 21) == 0
        primset{1, 1}.ori(ori_id, :) = prim_save(idx, 1:20);
        primset{1, 1}.cen(ori_id, :) = primset{1, 1}.ori(ori_id, 11:13)/2.0 + primset{1, 1}.ori(ori_id, 14:16);
        primset{1, 1}.side(ori_id) = 0;
        if sem_flag
            primset{1, 1}.cls(ori_id) = prim_save(idx, 22);
        end
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
        if sem_flag
            primset{1, 1}.cls(ori_id) = prim_save(idx, 22);
        end
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
[sem_bbox] = merge_semantic_bbox(prim_save, gt_num, sem_flag);
primset{1, 1}.sem = sem_bbox;
end

function [heights, height_idx] = change_idxs_in_rotation_order(heights, height_idx, prim_save, sem_flag)
rot_idx = [];
rot_h = [];
tmp_idx = height_idx(1);
tmp_heights = heights(1);
if sem_flag
    clses = prim_save(height_idx, 22);
else
    clses = 0*height_idx - 1;
end
for i = 2:size(heights, 1)
    con_1 = (abs(heights(i) - heights(i-1)) < 3 && (clses(i) == clses(i-1) && clses(i) ~= -1));
    con_2 = (prim_save(height_idx(i), 21) == -1) && (prim_save(height_idx(i-1), 21) == -1);
    con_3 = (clses(i) == 6 && clses(i-1) == 6);
    if con_3 %con_1 && con_2
        tmp_idx = [tmp_idx height_idx(i)];
        tmp_heights = [tmp_heights heights(i)];
    else
        if numel(tmp_idx) > 1
            [tmp_idx, tmp_heights] = idxs_in_order_rot(tmp_idx, tmp_heights, prim_save);
        end
        rot_idx = [rot_idx tmp_idx];
        rot_h = [rot_h tmp_heights];
        tmp_idx = height_idx(i);
        tmp_heights = heights(i);
    end
end
if numel(tmp_idx) > 1
    [tmp_idx, tmp_heights] = idxs_in_order_rot(tmp_idx, tmp_heights, prim_save);
end
height_idx = [rot_idx tmp_idx]';
heights = [rot_h tmp_heights]';
end

function [tmp_idx, tmp_heights] = idxs_in_order_rot(tmp_idx, tmp_heights, prim_save)
rots = prim_save(tmp_idx, 20);
[rots_sorted, rots_idx] = sort(rots, 'descend');
tmp_idx = tmp_idx(rots_idx);
tmp_heights = tmp_heights(rots_idx);
end

function [sem_bbox] = merge_semantic_bbox(prim_save, gt_num, sem_flag)
sample_grid = 7;
min_max = zeros(6, 6);
min_max(:, 1:3) = 1000;
if sem_flag
    num = size(prim_save, 1);
else
    num = 0;
end
for i = 1:num
    prim_i = prim_save(i, :);
    sem_tag = prim_i(1, 22);
    scale = [1,1,1];
    shape = prim_i(1,11:13)';
    trans = prim_i(1,14:16);
    Rv = prim_i(1,17:19)';
    [Rv, theta] = rv_to_rv_theta(Rv);
    % affine transformation
    vx = getVX(Rv);% rotation
    Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
    [sample_pt] = sample_primitive(shape, sample_grid);
    [~, sample_pt_dst] = get_mean_and_pt(sample_pt, trans, scale, Rrot);
    sample_pt_dst = min(sample_pt_dst, 32);
    sample_pt_dst = max(sample_pt_dst, 0);
    min_xyz = min(sample_pt_dst);
    max_xyz = max(sample_pt_dst);
    min_max(sem_tag, 1:3) = min(min_xyz, min_max(sem_tag, 1:3));
    min_max(sem_tag, 4:6) = max(max_xyz, min_max(sem_tag, 4:6));
end
xyz = min_max(:, 4:6) - min_max(:, 1:3);
sem_bbox = [min_max, xyz];
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