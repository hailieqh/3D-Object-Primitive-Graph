myvox = load(['../matlab/mat3dprnn/Myvox' num2str(3) '.mat']);
myvox = myvox.new_vox;
myprim = load(['../matlab/mat3dprnn/Myprim' num2str(3) '.mat']);
myprim = myprim.prim_save;

voxel_scale = 30;
gt_voxel = reshape(myvox.deleted(3,:,:,:), voxel_scale, voxel_scale, voxel_scale);
%gt_voxel = permute(gt_voxel, [3,2,1]);
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


prim_r = myprim(3, 1:20);
prim_pt_x = [0 prim_r(11) prim_r(11) 0 0 prim_r(11) prim_r(11) 0];
prim_pt_y = [0 0 prim_r(12) prim_r(12) 0 0 prim_r(12) prim_r(12)];
prim_pt_z = [0 0 0 0 prim_r(13) prim_r(13) prim_r(13) prim_r(13)];
prim_pt = [prim_pt_x' prim_pt_y' prim_pt_z'];
prim_pt = bsxfun(@plus, prim_pt, prim_r(:,14:16));
prim_pt_mean = mean(prim_pt);

Rv = prim_r(:,17:19);
vx = getVX(Rv);% rotation
theta = prim_r(:,20);
Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
prim_pt = bsxfun(@minus, prim_pt, prim_pt_mean);
prim_pt = prim_pt*Rrot;
prim_pt = bsxfun(@plus, prim_pt, prim_pt_mean);
prim_pt = bsxfun(@min,prim_pt,voxel_scale);
subplot(1,2,1)
[shape_vis]=VoxelPlotter(gt_voxel,1,1);
hold on; view(3); axis equal;scatter3(0,0,0);%%%
vertices = prim_pt;
faces = [1,2,3,4;5,6,7,8;1,2,6,5;3,4,8,7;1,4,8,5;2,3,7,6];
subplot(1,2,1);
patch('Faces',faces,'Vertices',vertices,'FaceColor',color{j},'FaceAlpha',0.3);
subplot(1,2,2);
light('Position',[-1 -1 0],'Style','local')
patch('Faces',faces,'Vertices',vertices,'FaceColor',color{j},'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.6, 'DiffuseStrength', 0.8, 'FaceAlpha',0.8);
view(3); axis equal; hold on;
axis([0,voxel_scale,0,voxel_scale,0,voxel_scale])

gt_prim = myprim(3, 1:20);
shape = gt_prim(1,11:13);
trans = gt_prim(1,14:16);
Rv = gt_prim(1,17:19);
theta = gt_prim(1,20);

sample_grid = 7;
sample_dist_x = shape(1)/sample_grid;% sampling 
sample_x = 0:sample_dist_x:shape(1);
sample_dist_y = shape(2)/sample_grid;
sample_y = 0:sample_dist_y:shape(2);
sample_dist_z = shape(3)/sample_grid;
sample_z = 0:sample_dist_z:shape(3);
[sample_X,sample_Y,sample_Z] = meshgrid(sample_x,sample_y,sample_z);
sample_pt = [sample_X(:), sample_Y(:), sample_Z(:)];
sample_pt_dst = bsxfun(@plus, sample_pt, trans);
cen_mean = mean(sample_pt_dst);

insum = 0;
for gt_i = 1:size(gt_pt, 1)
    [in_box, voxel] = check_voxel_in_box(gt_pt(gt_i, :), shape, trans, Rv, theta, cen_mean);
    insum = insum + in_box;
end
insum




%primset = save_primset(myprim, gt_num);
%primset
voxel_scale = 32;
dirlist = dir(fullfile('../pix3d/invoxel/'));
vox_count = 1;
for i = 1:size(dirlist,1)
    if i > 2 && i~= 7 && i ~= 14 && i ~= 20 && i ~= 22 && i ~= 28 && i ~= 29 && i ~= 35
    %if i > 2 && dirlist(i).name(1) == 'S'
        %myvox = load(['../pix3d/outvoxel/voxel' num2str(vox_count) '.mat']);
        myvox = load(['../pix3d/invoxel/' dirlist(i).name '/voxel.mat']);
        myvox = myvox.voxel;
        %myvox = load('/Users/heqian/Reaserch/projects/pix3d/pix3d/eval/baseline_output/0011.mat');
        %myvox = load('/Users/heqian/Reaserch/projects/pix3d/pix3d/model/chair/IKEA_REIDAR/voxel.mat');
        %myvox = load('/Users/heqian/Reaserch/projects/pix3d/pix3d/eval/voxel.mat');
        %myvox = myvox.voxel;
        %myvox = (myvox>0.1);
        myvox = load(['/Users/heqian/Reaserch/projects/pix3d/pix3d/eval/modelnet_chair.mat']);
        voxdata = myvox.voxTile;
        myvox = reshape(voxdata(vox_count, :,:,:),voxel_scale,voxel_scale,voxel_scale);
        %myvox = flip(myvox, 2);
        I = figure(1);
        subplot(1,2,1)
        [shape_vis]=VoxelPlotter(myvox,1,1);
        hold on; view(3); axis equal;scatter3(0,0,0);%%%
        out_volume = interpolate_volume(myvox);
        subplot(1,2,2)
        [shape_vis]=VoxelPlotter(out_volume,1,1);
        hold on; view(3); axis equal;scatter3(0,0,0);%%%
        %saveas(I, sprintf('../pix3d/imagevoxel/%d.jpg', vox_count));
        close(I);
        voxel = out_volume;
        voxelname = ['../pix3d/outvoxel/voxel' num2str(vox_count) '.mat'];
        %save(voxelname, 'voxel');
        voxTile(vox_count, :, :, :) = out_volume;
        %save('../pix3d/outvoxel/modelnet_chair.mat', 'voxTile');
        vox_count = vox_count + 1;
    end
end


function [out_volume] = interpolate_volume(in_volume)
[x,y,z] = size(in_volume);
out_volume = zeros(30,30,30);
a = 1;
vstart = 8;
vend = 1;
inter = 4;
for i = vstart:inter:(x-vend)
    b = 1;
    for j = vstart:inter:(y-vend)
        c = 1;
        for k = vstart:inter:(z-vend)
            out_volume(a, b, c) = in_volume(i, j, k);
            c = c + 1;
        end
        b = b + 1;
    end
    a = a + 1;
end

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

function [in_box, voxel] = check_voxel_in_box(voxel, shape, trans, Rv, theta, cen_mean)
in_box = 0;
voxel = voxel - 0.5;
voxel = bsxfun(@plus, voxel, -cen_mean);
theta = -theta;
vx = getVX(Rv);
Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
voxel = voxel * Rrot;
voxel = bsxfun(@minus, voxel, -cen_mean);
voxel = bsxfun(@plus, voxel, -trans);

% voxel = bsxfun(@plus, voxel, -trans);
% %cen_mean = mean(sample_pt_dst);
% voxel = bsxfun(@minus, voxel, -cen_mean);
% %voxel =  bsxfun(@times, voxel, scale);
% theta = -theta;
% vx = getVX(Rv);
% Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
% voxel = voxel * Rrot;
% voxel = bsxfun(@plus, voxel, -cen_mean);
thresh = 0.5;
if (voxel(1) >= 0 || abs(voxel(1)) < thresh) && (voxel(1) <= ceil(shape(1)) || abs(voxel(1) - shape(1)) < thresh)
    if (voxel(2) >= 0 || abs(voxel(2)) < thresh) && (voxel(2) <= ceil(shape(2)) || abs(voxel(2) - shape(2)) < thresh)
        if (voxel(3) >= 0 || abs(voxel(3)) < thresh) && (voxel(3) <= ceil(shape(3)) || abs(voxel(3) - shape(3)) < thresh)
            in_box = 1;
        end
    end
end
voxel = ceil(voxel);
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