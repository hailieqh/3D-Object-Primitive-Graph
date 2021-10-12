init = 0;
cls = '3dprnnnight_stand';
local_root = '/Users/heqian/Research/projects/primitive-based_3d';
root = [local_root '/process_data/all'];
output_dir = [root '/output/' cls];
edit_dir = [output_dir '/primset_sem'];
if init % 3dprnntable or 3dprnnnight_stand
    rot_vox_all(edit_dir, cls);
end
sem_flag = 1;
for i = 168:1:6000
    if exist([edit_dir '/Myprimset_all.mat'], 'file')
        load([edit_dir '/Myprimset_all.mat'], 'primset');
    end
    prims_i = primset{i, 1};
    for j = 1:numel(prims_i.side)
        if prims_i.side(j) == 1
            sym_tmp = prims_i.sym(j, :);
            prims_i.sym(j, :) = prims_i.ori(j, :);
            prims_i.ori(j, :) = sym_tmp;
            prims_i.side(j) = 0;
        end
    end
%     prims_i.cls = zeros(1, numel(prims_i.side));
    prims_i = order_in_cheight(prims_i);
    % checkpoint and edit
    primset{i, 1} = prims_i;
    save([edit_dir '/Myprimset_all.mat'], 'primset');
end

function [out] = rot_vox_all(edit_dir, cls)
voxel_scale = 30;
voxTile = load([edit_dir '/modelnet_all.mat']);
voxTile = voxTile.voxTile;
save_model_dir = [edit_dir '/modelnet_all.mat'];
for i = 1:1:size(voxTile, 1)
    voxel_down_i = reshape(voxTile(i,:,:,:),voxel_scale,voxel_scale,voxel_scale);
%     if i == 5 || i == 34 || i == 72 || i == 86 || i == 89 || i == 184 ...
%         || i == 218 || i == 230 || i == 236 || i == 240 || i == 267 ...
%         || i == 299 || i == 303 || i == 323 || i == 380 || i == 391 ...
%         || i == 409
%         k = 3;
% %         k = 1;
%     else
%         k = 3;
%     end
    k = 1;
    if 1%i ~= 17
        voxel_down_i = rot90(voxel_down_i, k);%rot along axis z
        if strcmp(cls, '3dprnntable')
            voxel_down_i = flip(voxel_down_i, 1);%flip along axis x
        end
        voxTile(i,:,:,:) = voxel_down_i;
    end
end
save(save_model_dir, 'voxTile');
out = 1;
end


function [prims_out] = order_in_cheight(prims_i)
prims_out = prims_i;
prims_i_ori = prims_i.ori;
xs = prims_i_ori(:, 11)/2.0 + prims_i_ori(:, 14);
ys = prims_i_ori(:, 12)/2.0 + prims_i_ori(:, 15);
zs = prims_i_ori(:, 13)/2.0 + prims_i_ori(:, 16);
prim_cheight = prims_i_ori(:, 13)/2.0 + prims_i_ori(:, 16);
[heights, height_idx] = sort(prim_cheight, 'descend');
for i = 1:numel(prims_i.side)
    j = height_idx(i);
    prims_out.ori(i, :) = prims_i.ori(j, :);
    prims_out.sym(i, :) = prims_i.sym(j, :);
    prims_out.cen(i, :) = [xs(j), ys(j), zs(j)];
    if heights(i) ~= zs(j)
        disp(i);
    end
end
end