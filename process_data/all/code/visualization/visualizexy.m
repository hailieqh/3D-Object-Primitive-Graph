% visualize gt primitive
addpath('./Voxel Plotter/');

cls =  'chair';
type = 'train';
visual = 1;

use_set = 1;
if use_set
    load('../box_generation/post_primset/Myprimset.mat', 'primset');
%     load('../box_generation/mat/Myprimset.mat', 'primset');
end

%load(['../data/prim_gt/prim_sort_mn_' cls '_' type '.mat'],'primset');
%load(['prim_sort_mn_nightstand_' type '.mat'],'primset');

%vox_data = load(['../data/ModelNet10_mesh/' cls '/modelnet_' cls '_' type '.mat']);
% vox_data = load(['../pix3d/modelnet_' cls '.mat']);
vox_data = load('/Users/xiangyi/qian/pix3d_to_prim/pix3d/eval/modelnet_chair.mat');
%vox_data = load(['../matlab/mat/Myvoxset.mat']);
vox_data = vox_data.voxTile;
%mesh_data = load(['../data/ModelNet10_mesh/' cls '/mesh_' type '.mat']);
%obj_mesh = mesh_data;

color = {'red', 'green', 'blue', 'cyan', 'yellow', 'magenta','black','black','white','white', 'red', 'green', 'blue', 'cyan', 'yellow', 'magenta','black','black','white'};

if use_set
    primset2  = load('../box_generation/post_primset_ori_wrong/Myprimset.mat');
end
primset2 = primset2.primset;

vox_data2 = load('/Users/xiangyi/qian/primitive-based_3d/2D_3D/modelnet_chair.mat');
vox_data2 = vox_data2.voxTile;
color = {'red', 'green', 'blue', 'cyan', 'yellow', 'magenta','black','black','white','white', 'red', 'green', 'blue', 'cyan', 'yellow', 'magenta','black','black','white'};


voxel_scale = 32; %30
for i = 210:216%1:numel(primset)
    disp(i);
    prim_num = i;
    vox_gt = reshape(vox_data(prim_num,:,:,:),voxel_scale,voxel_scale,voxel_scale);
    I = figure(1);
    
    subplot(1,3,1)
    title('Flip true');
    [shape_vis]=VoxelPlotter(vox_gt,1,1);
    hold on; view(3); axis equal;scatter3(0,0,0);
    subplot(1,3,2)
    [shape_vis]=VoxelPlotter(vox_gt,1,1);
    hold on; view(3); axis equal;scatter3(0,0,0);
    h3 = subplot(1,3,3);cla(h3);
    if ~use_set
        myvox = load(['../box_generation/post_primset/Myvox' num2str(prim_num) '.mat']);
        myvox = myvox.new_vox;
        myprim = load(['../box_generation/post_primset/Myprim' num2str(prim_num) '.mat']);
%         myprim = load(['../box_generation/post_primset/Myprim' num2str(prim_num) '.mat']);
        myprim = myprim.prim_save;
    end
%     myvox = load(['../matlab/mat/Myvox_refine' num2str(prim_num) '.mat']);
%     myvox = myvox.new_vox_refine;
%     myprim = load(['../matlab/mat/Myprim_refine' num2str(prim_num) '.mat']);
%     myprim = myprim.prim_save_refine;
    if use_set
        size_prim = size(primset{i}.ori,1);
    else
        size_prim = size(myprim,1);
    end
    for j = 1:size_prim
        %prim_r(1,14:20)=[0,30-prim_r(12),10,0,0,0,0];
        prim_idx = j;%%%
        if use_set
            prim_r = primset{i}.ori(j,:);
        else
            prim_r = myprim(prim_idx, 1:20);
            sem_tag = myprim(prim_idx, 22);
        end
        %prim_r = [zeros(1,10), 6.2130    8.6016    1.7356   20.8961   17.6983    7.6187         0      0  1  0.1800];
        %prim_r = [zeros(1,10), 16.9617    4.1199   21.1364    7.9954   21.0733   16.8673    1  0         0  0.1200];
        %prim_r(1, 14:20) = 0;
        if 0%~use_set
            myvox_gt = reshape(myvox.remained(prim_idx,:,:,:),voxel_scale,voxel_scale,voxel_scale);
            subplot(1,3,2)
            [shape_vis]=VoxelPlotter(myvox_gt,1,1);
            hold on; view(3); axis equal;scatter3(0,0,0);%%%
        end
        
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

        if use_set
            color_tag = j;
        else
            color_tag = sem_tag;
        end
        vertices = prim_pt;
        faces = [1,2,3,4;5,6,7,8;1,2,6,5;3,4,8,7;1,4,8,5;2,3,7,6];
        subplot(1,3,2);
        patch('Faces',faces,'Vertices',vertices,'FaceColor',color{color_tag},'FaceAlpha',0.3);
        subplot(1,3,3);
        light('Position',[-1 -1 0],'Style','local')
        patch('Faces',faces,'Vertices',vertices,'FaceColor',color{color_tag},'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.6, 'DiffuseStrength', 0.8, 'FaceAlpha',0.8);
        view(3); axis equal; hold on;
        axis([0,voxel_scale,0,voxel_scale,0,voxel_scale])
        %figure(2)
        %light('Position',[5 100 15],'Style','local')
                %patch('Faces',faces,'Vertices',vertices,'FaceColor',color{tol_cnt},'FaceAlpha',0.8);
        %        patch('Faces',faces,'Vertices',vertices,'FaceColor',[0.75 0.75 0.75],'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.8, 'DiffuseStrength', 0.3, 'FaceAlpha',1);
        %        view(3); axis equal; hold on;
                %figure(1)
        if ~use_set
            continue%%%heqian
        end
        prim_r = primset{i}.sym(j,:);
        
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
        
        vertices = prim_pt;
        faces = [1,2,3,4;5,6,7,8;1,2,6,5;3,4,8,7;1,4,8,5;2,3,7,6];
        subplot(1,3,2);
        patch('Faces',faces,'Vertices',vertices,'FaceColor',color{color_tag},'FaceAlpha',0.3);
        subplot(1,3,3);
        light('Position',[-1 -1 0],'Style','local')
        patch('Faces',faces,'Vertices',vertices,'FaceColor',color{color_tag},'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.6, 'DiffuseStrength', 0.8, 'FaceAlpha',0.8);
        view(3); axis equal; hold on;
        axis([0,voxel_scale,0,voxel_scale,0,voxel_scale])
        %figure(2)
        %light('Position',[5 100 15],'Style','local')
                %patch('Faces',faces,'Vertices',vertices,'FaceColor',color{tol_cnt},'FaceAlpha',0.8);
        %        patch('Faces',faces,'Vertices',vertices,'FaceColor',[0.75 0.75 0.75],'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.8, 'DiffuseStrength', 0.3, 'FaceAlpha',1);
        %        view(3); axis equal; hold on;
                figure(1)               
    end
     %keyboard
%     saveas(I, sprintf('../fitting_sem/%d.jpg', i));
    disp(i);
    prim_num = i;
    vox_gt = reshape(vox_data2(prim_num,:,:,:),voxel_scale,voxel_scale,voxel_scale);
     
    
%     subplot(2,3,4)
%     title('Origin wrong');
%     [shape_vis]=VoxelPlotter(vox_gt,1,1);
%     hold on; view(3); axis equal;scatter3(0,0,0);
%     subplot(2,3,5)
%     [shape_vis]=VoxelPlotter(vox_gt,1,1);
%     hold on; view(3); axis equal;scatter3(0,0,0);
%     h3 = subplot(2,3,6);cla(h3);
    if ~use_set
        myvox = load(['../box_generation/post_primset/Myvox' num2str(prim_num) '.mat']);
        myvox = myvox.new_vox;
        myprim = load(['../box_generation/post_primset/Myprim' num2str(prim_num) '.mat']);
%         myprim = load(['../box_generation/post_primset/Myprim' num2str(prim_num) '.mat']);
        myprim = myprim.prim_save;
    end
%     myvox = load(['../matlab/mat/Myvox_refine' num2str(prim_num) '.mat']);
%     myvox = myvox.new_vox_refine;
%     myprim = load(['../matlab/mat/Myprim_refine' num2str(prim_num) '.mat']);
%     myprim = myprim.prim_save_refine;




%     if use_set
%         size_prim = size(primset2{i}.ori,1);
%     else
%         size_prim = size(myprim,1);
%     end
%     for j = 1:size_prim
%         %prim_r(1,14:20)=[0,30-prim_r(12),10,0,0,0,0];
%         prim_idx = j;%%%
%         if use_set
%             prim_r = primset2{i}.ori(j,:);
%         else
%             prim_r = myprim(prim_idx, 1:20);
%             sem_tag = myprim(prim_idx, 22);
%         end
%         %prim_r = [zeros(1,10), 6.2130    8.6016    1.7356   20.8961   17.6983    7.6187         0      0  1  0.1800];
%         %prim_r = [zeros(1,10), 16.9617    4.1199   21.1364    7.9954   21.0733   16.8673    1  0         0  0.1200];
%         %prim_r(1, 14:20) = 0;
%         if 0%~use_set
%             myvox_gt = reshape(myvox.remained(prim_idx,:,:,:),voxel_scale,voxel_scale,voxel_scale);
%             subplot(1,3,2)
%             [shape_vis]=VoxelPlotter(myvox_gt,1,1);
%             hold on; view(3); axis equal;scatter3(0,0,0);%%%
%         end
%         
%         prim_pt_x = [0 prim_r(11) prim_r(11) 0 0 prim_r(11) prim_r(11) 0];
%         prim_pt_y = [0 0 prim_r(12) prim_r(12) 0 0 prim_r(12) prim_r(12)];
%         prim_pt_z = [0 0 0 0 prim_r(13) prim_r(13) prim_r(13) prim_r(13)];
%         prim_pt = [prim_pt_x' prim_pt_y' prim_pt_z'];
%         prim_pt = bsxfun(@plus, prim_pt, prim_r(:,14:16));
%         prim_pt_mean = mean(prim_pt);
%     
%         Rv = prim_r(:,17:19);
%         vx = getVX(Rv);% rotation
%         theta = prim_r(:,20);
%         Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
%         prim_pt = bsxfun(@minus, prim_pt, prim_pt_mean);
%         prim_pt = prim_pt*Rrot;
%         prim_pt = bsxfun(@plus, prim_pt, prim_pt_mean);
%         prim_pt = bsxfun(@min,prim_pt,voxel_scale);
% 
%         if use_set
%             color_tag = j;
%         else
%             color_tag = sem_tag;
%         end
%         vertices = prim_pt;
%         faces = [1,2,3,4;5,6,7,8;1,2,6,5;3,4,8,7;1,4,8,5;2,3,7,6];
%         subplot(2,3,5);
%         patch('Faces',faces,'Vertices',vertices,'FaceColor',color{color_tag},'FaceAlpha',0.3);
%         subplot(2,3,6);
%         light('Position',[-1 -1 0],'Style','local')
%         patch('Faces',faces,'Vertices',vertices,'FaceColor',color{color_tag},'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.6, 'DiffuseStrength', 0.8, 'FaceAlpha',0.8);
%         view(3); axis equal; hold on;
%         axis([0,voxel_scale,0,voxel_scale,0,voxel_scale])
%         %figure(2)
%         %light('Position',[5 100 15],'Style','local')
%                 %patch('Faces',faces,'Vertices',vertices,'FaceColor',color{tol_cnt},'FaceAlpha',0.8);
%         %        patch('Faces',faces,'Vertices',vertices,'FaceColor',[0.75 0.75 0.75],'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.8, 'DiffuseStrength', 0.3, 'FaceAlpha',1);
%         %        view(3); axis equal; hold on;
%                 %figure(1)
%         if ~use_set
%             continue%%%heqian
%         end
%         prim_r = primset2{i}.sym(j,:);
%         
%         prim_pt_x = [0 prim_r(11) prim_r(11) 0 0 prim_r(11) prim_r(11) 0];
%         prim_pt_y = [0 0 prim_r(12) prim_r(12) 0 0 prim_r(12) prim_r(12)];
%         prim_pt_z = [0 0 0 0 prim_r(13) prim_r(13) prim_r(13) prim_r(13)];
%         prim_pt = [prim_pt_x' prim_pt_y' prim_pt_z'];
%         prim_pt = bsxfun(@plus, prim_pt, prim_r(:,14:16));
%         prim_pt_mean = mean(prim_pt);
%     
%         Rv = prim_r(:,17:19);
%         vx = getVX(Rv);% rotation
%         theta = prim_r(:,20);
%         Rrot = cos(theta)*eye(3) + sin(theta)*vx + (1-cos(theta))*Rv'*Rv;
%         prim_pt = bsxfun(@minus, prim_pt, prim_pt_mean);
%         prim_pt = prim_pt*Rrot; 
%         prim_pt = bsxfun(@plus, prim_pt, prim_pt_mean);
%         prim_pt = bsxfun(@min,prim_pt,voxel_scale);
%         
%         vertices = prim_pt;
%         faces = [1,2,3,4;5,6,7,8;1,2,6,5;3,4,8,7;1,4,8,5;2,3,7,6];
%         subplot(2,3,5);
%         patch('Faces',faces,'Vertices',vertices,'FaceColor',color{color_tag},'FaceAlpha',0.3);
%         subplot(2,3,6);
%         light('Position',[-1 -1 0],'Style','local')
%         patch('Faces',faces,'Vertices',vertices,'FaceColor',color{color_tag},'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.6, 'DiffuseStrength', 0.8, 'FaceAlpha',0.8);
%         view(3); axis equal; hold on;
%         axis([0,voxel_scale,0,voxel_scale,0,voxel_scale])
%         %figure(2)
%         %light('Position',[5 100 15],'Style','local')
%                 %patch('Faces',faces,'Vertices',vertices,'FaceColor',color{tol_cnt},'FaceAlpha',0.8);
%         %        patch('Faces',faces,'Vertices',vertices,'FaceColor',[0.75 0.75 0.75],'EdgeColor','none', 'FaceLighting', 'gouraud', 'AmbientStrength',0.8, 'DiffuseStrength', 0.3, 'FaceAlpha',1);
%         %        view(3); axis equal; hold on;
%                 figure(1)               
%     end
%      keyboard
% %     saveas(I, sprintf('../fitting_sem/%d.jpg', i));
%     save_file = 'images_flip_new/';
%     if ~exist([save_file cls],'dir')==1
%        mkdir([save_file cls]);
%     end
%     saveas(I, sprintf([save_file cls '/' num2str(i) '.jpg']));

    close(I);
  
end








