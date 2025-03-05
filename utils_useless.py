import torch
import torch.nn.functional as F
import numpy as np
import random
import time
from copy import deepcopy
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg


"""
Define task metrics, loss functions and model trainer here.
"""

def control_seed(seed):
    torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_fit(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == 'semantic':
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == 'depth':
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    if task_type == 'normal':
        # normal loss: dot product
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)

    return loss

# Legacy: compute mIoU and Acc. for each image and average across all images.

# def compute_miou(x_pred, x_output):
#     _, x_pred_label = torch.max(x_pred, dim=1)
#     x_output_label = x_output
#     batch_size = x_pred.size(0)
#     class_nb = x_pred.size(1)
#     device = x_pred.device
#     for i in range(batch_size):
#         true_class = 0
#         first_switch = True
#         invalid_mask = (x_output[i] >= 0).float()
#         for j in range(class_nb):
#             pred_mask = torch.eq(x_pred_label[i], j * torch.ones(x_pred_label[i].shape).long().to(device))
#             true_mask = torch.eq(x_output_label[i], j * torch.ones(x_output_label[i].shape).long().to(device))
#             mask_comb = pred_mask.float() + true_mask.float()
#             union = torch.sum((mask_comb > 0).float() * invalid_mask)  # remove non-defined pixel predictions
#             intsec = torch.sum((mask_comb > 1).float())
#             if union == 0:
#                 continue
#             if first_switch:
#                 class_prob = intsec / union
#                 first_switch = False
#             else:
#                 class_prob = intsec / union + class_prob
#             true_class += 1
#         if i == 0:
#             batch_avg = class_prob / true_class
#         else:
#             batch_avg = class_prob / true_class + batch_avg
#     return batch_avg / batch_size
#
#
# def compute_iou(x_pred, x_output):
#     _, x_pred_label = torch.max(x_pred, dim=1)
#     x_output_label = x_output
#     batch_size = x_pred.size(0)
#     for i in range(batch_size):
#         if i == 0:
#             pixel_acc = torch.div(
#                 torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).float()),
#                 torch.sum((x_output_label[i] >= 0).float()))
#         else:
#             pixel_acc = pixel_acc + torch.div(
#                 torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).float()),
#                 torch.sum((x_output_label[i] >= 0).float()))
#     return pixel_acc / batch_size


# New mIoU and Acc. formula: accumulate every pixel and average across all pixels in all images
class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu), acc


def depth_error(x_pred, x_output):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_true = x_pred.masked_select(binary_mask)
    x_output_true = x_output.masked_select(binary_mask)
    abs_err = torch.abs(x_pred_true - x_output_true)
    rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
    return (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item(), \
           (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()


def normal_error(x_pred, x_output):
    binary_mask = (torch.sum(x_output, dim=1) != 0)
    error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)



def compute_sdf01(segmentation):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM)
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation

    """
    # print(type(segmentation), segmentation.shape)

    segmentation = segmentation.astype(np.uint8)
    if len(segmentation.shape) == 4:  # 3D image
        segmentation = np.expand_dims(segmentation, 1)
    normalized_sdf = np.zeros(segmentation.shape)
    if segmentation.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(segmentation.shape[0]):  # batch size
        for c in range(dis_id, segmentation.shape[1]):  # class_num
            # ignore background
            posmask = segmentation[b][c]
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = negdis / np.max(negdis) / 2 - posdis / np.max(posdis) / 2 + 0.5
            sdf[boundary > 0] = 0.5
            normalized_sdf[b][c] = sdf
    return normalized_sdf

def compute_sdf1_1(segmentation):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, class, x, y, z)
    output: the Signed Distance Map (SDM)
    sdm(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation

    """
    # print(type(segmentation), segmentation.shape)

    segmentation = segmentation.astype(np.uint8)
    if len(segmentation.shape) == 4: # 3D image
        segmentation = np.expand_dims(segmentation, 1)
    normalized_sdf = np.zeros(segmentation.shape)
    if segmentation.shape[1] == 1:
        dis_id = 0
    else:
        dis_id = 1
    for b in range(segmentation.shape[0]): # batch size
        for c in range(dis_id, segmentation.shape[1]): # class_num
            # ignore background
            posmask = segmentation[b][c]
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = negdis/np.max(negdis) - posdis/np.max(posdis)
            sdf[boundary>0] = 0
            normalized_sdf[b][c] = sdf
    return normalized_sdf