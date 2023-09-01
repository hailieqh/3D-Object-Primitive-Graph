import scipy.io
import os
import random
import copy
import torch
from skimage import io, transform
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from lib.opts import *


# Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
# First value to be returned is average accuracy across 'idxs', followed by individual accuracies
def accuracy(output, label, std):
    # output (6, 10, 64, 64)
    # label (6, 10, 64, 64)
    # label_xy (6, 10, 2)
    global opt
    opt = get_opt()
    batch_size0, num_kp0, h, w = output.size()
    batch_size1, num_kp1, h, w = label.size()
    assert batch_size0 == batch_size1 and num_kp0 == num_kp1
    l0, l1, l2 = batch_size0, num_kp0, 2
    num_pixel = h * w
    output_xy = map_to_point(output)  # (6, 10, 2)
    label_xy = map_to_point(label)  # (6, 10, 2)
    normalize_pck = torch.ones(l0) * w / 10.0  # (6) -- 0.1max(h,w)
    normalize_pcp = std
    dists = compute_dist(output_xy, label_xy)
    pck_thr = 0.5
    if opt.dataset == 'car':
        pck_thr *= 2
    pcp_thr = opt.PCPthr  # 0.2*torso # 1.5*std
    ae_thr = 5
    acc_pck = compute_acc(dists, pck_thr, 'PCK', normalize_pck)
    acc_pcp = compute_acc(dists, pcp_thr, 'PCP', normalize_pcp)
    ae = compute_ae(dists, ae_thr, normalize_pcp)
    return (acc_pck, acc_pcp, ae)  # accuracy of all points, accuracy of every point..


def compute_ae(dists, threshold, normalize_pcp):
    l0, l1 = dists.size()
    thr = torch.ones(l0, l1) * threshold
    dists = torch.ne(dists, -1).float() * dists / normalize_pcp
    err = torch.mean(torch.min(dists, thr))
    return err


def compute_acc(dists, threshold, metric, normalize):
    l0, l1 = dists.size()
    acc = {}
    avgAcc = 0.0
    badIdxCount = 0
    for i in range(l1):
        tmp = copy.deepcopy(dists[:, i])
        tmp1 = copy.deepcopy(dists[:, i])
        if metric == 'PCK':
            tmp /= normalize
        elif metric == 'PCP':
            tmp /= normalize[:, i]
        if torch.ne(tmp1, -1).sum() > 0:
            # acc -- average accuracy among a batch of images on one keypoint
            acc[i + 1] = tmp.le(threshold).eq(tmp1.ne(-1)).sum() / tmp1.ne(-1).sum()
            avgAcc = avgAcc + acc[i + 1]
        else:
            acc[i + 1] = -1
            badIdxCount += 1
    if l1 == badIdxCount:
        acc[0] = 1
    else:
        acc[0] = avgAcc / (l1 - badIdxCount)
    return acc


def compute_dist(output_xy, label_xy):
    l0, l1, l2 = output_xy.size()
    dists = torch.zeros(l0, l1)
    for i in range(l0):
        for j in range(l1):
            if label_xy[i][j][0] > 0 and label_xy[i][j][1] > 0:
                dists[i][j] = torch.dist(output_xy[i][j], label_xy[i][j])
                # if metric == 'PCK':
                #   dists[i][j] /= normalize_pck[i]
            else:
                dists[i][j] = -1
    return dists


def map_to_point(heatmap):
    batch_size, num_kp, h, w = heatmap.size()
    l0, l1, l2 = batch_size, num_kp, 2
    num_pixel = h * w
    heatmap = heatmap.view(l0, l1, -1)
    omaxprob, omaxidx = torch.max(heatmap, 2)  # (6, 10, 1)
    output_xy = torch.zeros(l0, l1, l2)  # (6, 10, 2)
    for i in range(l0):
        for j in range(l1):
            # import pdb;pdb.set_trace()
            output_xy[i][j][0] = omaxidx[i][j].item() % w
            output_xy[i][j][1] = omaxidx[i][j].item() / w
            # output_xy[i][j][0] = omaxidx[i][j][0].data[0] % w
            # output_xy[i][j][1] = omaxidx[i][j][0].data[0] / w
    return output_xy


def map_to_point_local(heatmap):
    batch_size, num_class, h, w = heatmap.size()
    # num_kp = opt.nKeypoints
    threshold = 0.05
    windowsize = 7
    aboveid = heatmap > threshold
    for b in range(batch_size):
        for c in range(num_class):
            for x in range(w - windowsize + 1):
                for y in range(h - windowsize + 1):
                    if torch.sum(aboveid[x:x + windowsize, y:y + windowsize]).data[0] == windowsize * windowsize:
                        window = heatmap[b, c, x:x + windowsize, y:y + windowsize]
                        omaxprob, omaxidx = torch.max(window, dim=0)
                        output_x = omaxidx.data[0] % w
                        output_y = omaxidx.data[0] / w


def flip(img):
    img = img.data[0].cpu().numpy()
    img = img[:, :, ::-1]
    c, h, w = img.shape
    out = np.zeros((1, c, h, w))
    out[0] = img
    out = torch.from_numpy(out.copy())
    return out


def shuffleLR(label):
    return label
    label = label.numpy()
    out = np.zeros(label.shape)
    global opt
    opt = get_opt()
    # n = opt.nKeypoints
    n = label.shape[1]
    nn = int(n / 2)
    if opt.dataset == 'bed' or opt.dataset == 'sofa' or opt.dataset == 'car':
        out[:, 0:nn, :, :] = copy.deepcopy(label[:, nn:n, :, :])
        out[:, nn:n, :, :] = copy.deepcopy(label[:, 0:nn, :, :])
    if opt.dataset == 'chair' or opt.dataset == 'table':
        for i in xrange(nn):
            j = i * 2
            tmp = copy.deepcopy(out[:, j, :, :])
            out[:, j, :, :] = copy.deepcopy(out[:, j + 1, :, :])
            out[:, j + 1, :, :] = copy.deepcopy(tmp)
    if opt.dataset == 'swivelchair':
        inter = [4, 2]
        for i in xrange(2):
            tmp = copy.deepcopy(out[:, i, :, :])
            out[:, i, :, :] = copy.deepcopy(out[:, i + inter[i], :, :])
            out[:, i + inter[i], :, :] = copy.deepcopy(tmp)
        for i in xrange(3):
            j = i * 2 + 7
            tmp = copy.deepcopy(out[:, j, :, :])
            out[:, j, :, :] = copy.deepcopy(out[:, j + 1, :, :])
            out[:, j + 1, :, :] = copy.deepcopy(tmp)
    if opt.dataset == 'flic':
        for i in xrange(3):
            tmp = copy.deepcopy(out[:, i, :, :])
            out[:, i, :, :] = copy.deepcopy(out[:, i + 3, :, :])
            out[:, i + 3, :, :] = copy.deepcopy(tmp)
        for i in xrange(2):
            j = i * 2 + 6
            tmp = copy.deepcopy(out[:, j, :, :])
            out[:, j, :, :] = copy.deepcopy(out[:, j + 1, :, :])
            out[:, j + 1, :, :] = copy.deepcopy(tmp)
    out = torch.from_numpy(out)
    return out


def show_heatmap(array, row):
    array = array.data.cpu().numpy()
    batch, num_kp, h, w = array.shape
    outmap = copy.deepcopy(array[0, 0, :, :])
    for i in range(1, num_kp):
        outmap += array[0, i, :, :]
    plt.imshow(outmap)
    show_pt_heatmap(array, num_kp, row)


def show_pt_heatmap(array, num_kp, row):
    for i in range(num_kp):
        ax = plt.subplot(3, num_kp, i + 1 + row * num_kp)
        ax.axis('off')
        plt.imshow(array[0, i, :, :])


def visualize(image, label, output, i, label_kp=None):
    global opt
    opt = get_opt()
    image = image.data.cpu().numpy()
    image = image[0].transpose((1, 2, 0))
    if label_kp is None:
        label_pt = map_to_point(label)
    else:
        label_pt = label_kp
    output_pt = map_to_point(output)
    label_pt = label_pt.cpu().numpy()
    output_pt = output_pt.cpu().numpy()
    label_pt *= 256.0 / 64.0
    output_pt *= 256.0 / 64.0
    # print(label_pt, output_pt)
    # plt.ion()
    fig = plt.figure()
    ax = plt.subplot(3, 4, 1)
    # plt.tight_layout()
    ax.set_title('image&gt_pt')
    show_labels(image, label_pt)

    ax = plt.subplot(3, 4, 2)
    ax.set_title('image&pred_pt')
    show_labels(image, output_pt)

    ax = plt.subplot(3, 4, 3)
    ax.set_title('image&gt_map')
    show_heatmap(label, 1)

    ax = plt.subplot(3, 4, 4)
    ax.set_title('image&pred_map')
    show_heatmap(output, 2)

    plt.savefig('out' + opt.env + '/{}.png'.format(i))
    # plt.show()
    plt.close('all')


def visualize_img_1(image, label, i):
    fig = plt.figure()
    plt.imshow(image)
    plt.scatter(label[:, 0], label[:, 1], s=100, marker='.', c='r')
    plt.savefig('out' + opt.env + '/1111{}.png'.format(i))
    plt.close('all')


def visualize_img_2(image, label, i):
    image = image.numpy().transpose((1, 2, 0))
    label = label.numpy() * 256.0 / 64.0
    fig = plt.figure()
    plt.imshow(image)
    plt.scatter(label[:, 0], label[:, 1], s=100, marker='.', c='r')
    plt.savefig('out' + opt.env + '/2222{}.png'.format(i))
    plt.close('all')


def show_labels(image, label):
    plt.imshow(image)
    plt.scatter(label[0, :, 0], label[0, :, 1], s=100, marker='.', c='r')


def plot_curve(loss, PCK, PCP, AE):
    plt.subplot(2, 2, 1)
    plt.plot(loss['train'], label='train')
    plt.plot(loss['val'], label='val')
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 2, 2)
    plt.plot(PCK['train'], label='train')
    plt.plot(PCK['val'], label='val')
    plt.title('PCK accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('PCK accuracy')

    plt.subplot(2, 2, 3)
    plt.plot(PCP['train'], label='train')
    plt.plot(PCP['val'], label='val')
    plt.title('PCP accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('PCP accuracy')

    plt.subplot(2, 2, 4)
    plt.plot(AE['train'], label='train')
    plt.plot(AE['val'], label='val')
    plt.title('AE history')
    plt.xlabel('Epoch')
    plt.ylabel('AE')

    plt.savefig('out' + opt.env + '/result.png')
    plt.close('all')


def random_init(seed=0):
    """Set the seed for the random for torch and random package
    Args:
        seed: random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
