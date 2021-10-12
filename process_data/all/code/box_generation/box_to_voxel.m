voxel_scale = 32;
sample_grid = 7;
load('./post_primset/Myprimset.mat','primset');
% load(['../data/prim_gt/prim_sort_mn_chair_train.mat'],'primset');
% load('./post_primset/Myprimset.mat','primset');
% load('./mat/Myprimset.mat','primset');
% primset = load('./mat/Myprimset_refine.mat');
% primset = primset.primset_refine;
% primset0 = load('./matrefine/Myprimset.mat');
% primset2 = load('./matrefine2/Myprimset.mat');
% primset{1} = primset0.primset{1};
% for i = 2:15
%     primset{i} = primset2.primset{i};
% end
%primset{16} = 
num = size(primset,1);
voxTile = zeros(num, voxel_scale, voxel_scale, voxel_scale);

for i = 1:num
    voxel = zeros(voxel_scale, voxel_scale, voxel_scale);
    for j = 1:size(primset{i}.ori,1)
        prim_r = primset{i}.ori(j,:);
        [voxel] = prim_to_voxel(voxel, prim_r, sample_grid, voxel_scale);
        prim_r = primset{i}.sym(j,:);
        [voxel] = prim_to_voxel(voxel, prim_r, sample_grid, voxel_scale);
    end
    voxTile(i, :, :, :) = voxel;
end

save('./post_primset/Myvoxset.mat','voxTile');


function [voxel] = prim_to_voxel(voxel, prim_r, sample_grid, voxel_scale)
shape = prim_r(1,11:13)';
trans = prim_r(1,14:16);
Rv = prim_r(1,17:19);
theta = prim_r(1,20);
scale = [1,1,1];
vx = getVX(Rv);% rotation
Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
[cen_mean, ~] = get_mean_and_pt(sample_grid, shape, trans, scale, Rrot);
for vx = 1:voxel_scale
    for vy = 1:voxel_scale
        for vz = 1:voxel_scale
            point = [vx, vy, vz];
            [in_box, ~] = check_voxel_in_box(point, shape, trans, Rv, theta, cen_mean);
            if in_box
                voxel(vx, vy, vz) = 1;
            end
        end
    end
end
end


function [cen_mean, sample_pt_dst] = get_mean_and_pt(sample_grid, shape, trans, scale, Rrot)
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
sample_pt_dst = bsxfun(@minus, sample_pt_dst, cen_mean);
sample_pt_dst = bsxfun(@times, sample_pt_dst, scale);
sample_pt_dst = sample_pt_dst * Rrot;
sample_pt_dst = bsxfun(@plus, sample_pt_dst, cen_mean);
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

thresh = 0.2;
if (voxel(1) >= 0 || abs(voxel(1)) < thresh) && (voxel(1) <= ceil(shape(1)) || abs(voxel(1) - shape(1)) < thresh)
    if (voxel(2) >= 0 || abs(voxel(2)) < thresh) && (voxel(2) <= ceil(shape(2)) || abs(voxel(2) - shape(2)) < thresh)
        if (voxel(3) >= 0 || abs(voxel(3)) < thresh) && (voxel(3) <= ceil(shape(3)) || abs(voxel(3) - shape(3)) < thresh)
            in_box = 1;
        end
    end
end
voxel = ceil(voxel);
end