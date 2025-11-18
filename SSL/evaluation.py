# -*- coding:utf-8 -*-

import torch
from torch import nn
import os
import numpy as np
from medmnist import INFO, Evaluator
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import math
from PIL import Image

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def norm(x):
    return F.normalize(x, dim=-1, eps=1e-10)

def save_img_att(original_img_array, map_array_resize,save_att_imgname):
    # original_imgs_array: (W,H,Channel)
    # map_array_resize: (c,w,h)

    c,w,h = map_array_resize.shape
    print('map_array_resize:',c,w,h)
    print('original_img_array:', original_img_array.shape)

    rgb_c = original_img_array.shape[2]

    if original_img_array.shape[2] == 1:
        original_img_array = original_img_array[:, :, 0]

    # Create the figure and subplots
    fig, axes = plt.subplots(c, 3, figsize=(12, 4*c))

    if c == 1:
        # Subplot 1: Original image
        if rgb_c == 1:
            axes[0].imshow(original_img_array, cmap='gray')
        else:
            axes[0].imshow(original_img_array)

        axes[0].axis('off')
        axes[0].set_title('Original Image')

        # Subplot 2: Attention map
        sns.heatmap(map_array_resize[0, :, :], ax=axes[1], cmap='viridis', cbar=False)
        axes[1].axis('off')
        if 'global' in save_att_imgname:
            axes[1].set_title('Global Attention Map')
        else:
            axes[1].set_title('Attention Map Class {}'.format(0))

        # Subplot 3: Heatmap overlay
        if rgb_c == 1:
            axes[2].imshow(original_img_array, cmap='gray')
        else:
            axes[2].imshow(original_img_array)
        sns.heatmap(map_array_resize[0, :, :], ax=axes[2], cmap='viridis', alpha=0.5, cbar=True)
        axes[2].axis('off')
        if 'global' in save_att_imgname:
            axes[2].set_title('Global Heatmap Overlay')
        else:
            axes[2].set_title('Heatmap Overlay Class {}'.format(0))

    else:
        for cls_id in range(c):

            # Subplot 1: Original image
            if rgb_c == 1:
                axes[cls_id,0].imshow(original_img_array, cmap='gray')
            else:
                axes[cls_id,0].imshow(original_img_array)
            # axes[cls_id,0].imshow(original_img_array)
            axes[cls_id,0].axis('off')
            axes[cls_id,0].set_title('Original Image')

            # Subplot 2: Attention map
            sns.heatmap(map_array_resize[cls_id,:,:], ax=axes[cls_id,1], cmap='viridis', cbar=False)
            axes[cls_id,1].axis('off')
            if 'global' in save_att_imgname:
                axes[cls_id,1].set_title('Global Attention Map')
            else:
                axes[cls_id,1].set_title('Attention Map Class {}'.format(cls_id))

            # Subplot 3: Heatmap overlay
            if rgb_c == 1:
                axes[cls_id, 2].imshow(original_img_array, cmap='gray')
            else:
                axes[cls_id, 2].imshow(original_img_array)
            sns.heatmap(map_array_resize[cls_id,:,:], ax=axes[cls_id,2], cmap='viridis', alpha=0.5, cbar=True)
            axes[cls_id,2].axis('off')
            if 'global' in save_att_imgname:
                axes[cls_id,2].set_title('Global Heatmap Overlay')
            else:
                axes[cls_id,2].set_title('Heatmap Overlay Class {}'.format(cls_id))

    # Adjust layout and add labels below subplots
    fig.tight_layout(pad=3)

    # Save the combined figure
    plt.savefig(save_att_imgname, bbox_inches='tight', dpi=300)
    plt.close()

def test_heatmap(test_dataloader, model, SAVE_DIR_Prefix):
    test_dataset = test_dataloader.dataset
    num_cls = test_dataset.num_cls
    # map_size = test_dataset.map_size
    original_imgs_array = test_dataset.imgs
    original_targets_array = test_dataset.targets
    sampling_num_per_cls = 2

    rgb_ch = test_dataset.img_channel

    for cls_id in range(num_cls):
        select_id = np.where(np.array(original_targets_array.squeeze(1)) == cls_id)[0].tolist()[:sampling_num_per_cls]
        for sample_id in select_id:
            original_img_array = original_imgs_array[sample_id]  # (W,H,Channel)
            data = original_img_array
            if rgb_ch != 3:
                data = np.repeat(original_img_array, 3, axis=2)

            w, h, c = data.shape

            if 'DINO_' in test_dataset.model_name or 'dinov2' in test_dataset.model_name:
                if test_dataset.img_resize == test_dataset.img_size:
                    data = Image.fromarray(data)
                    data = test_dataset.pre_process(data)

            data = data.unsqueeze(0)

            label = test_dataset.targets[sample_id].astype(int)

            data = data.cuda()

            embedding, embedding_mean,logits = model.forward_features(data)

            if num_cls == 2:
                check_gt = (logits[0].detach().cpu().numpy()>0)*1
                check_gt = check_gt[0]
            else:
                check_gt = np.argmax(logits[0].detach().cpu().numpy(), axis=-1)

            if 'DINO_vit' in model.backbone:
                embedding = embedding[:,1:,:]

            B_ALL, NUM_P, NUM_C = embedding.size()
            map_size = int(math.sqrt(NUM_P))

            cos_map = torch.einsum("bnc,bmc->bnm", norm(embedding), norm(embedding_mean))  # (B,N,1)
            cos_map = cos_map[0].detach().cpu()
            global_map_tensor_resize = F.interpolate(cos_map.view(1, 1, map_size, map_size), size=w,
                                                     mode='bilinear')  # (1,1,w,h)
            global_map_array_resize = global_map_tensor_resize.numpy()

            save_global_att_imgname = SAVE_DIR_Prefix + 'cls{}_sample{}_checkid{}_predid{}_global_att.png'.format(cls_id, sample_id,label,
                                                                                                  check_gt)
            save_img_att(original_img_array, global_map_array_resize[0], save_global_att_imgname)

            # no norm
            cos_map = torch.einsum("bnc,bmc->bnm", embedding, embedding_mean)  # (B,N,1)
            cos_map = cos_map[0].detach().cpu()
            global_map_tensor_resize = F.interpolate(cos_map.view(1, 1, map_size, map_size), size=w,
                                                     mode='bilinear')  # (1,1,w,h)
            global_map_array_resize = global_map_tensor_resize.numpy()

            save_global_att_imgname = SAVE_DIR_Prefix + 'cls{}_sample{}_checkid{}_predid{}_global_att_nonorm.png'.format(
                cls_id, sample_id, label,
                check_gt)
            save_img_att(original_img_array, global_map_array_resize[0], save_global_att_imgname)

def activation_visualize(map_size,local_map_all_tensor,global_map_all_tensor,y_true_array,original_imgs_array,original_targets_array,sampling_num_per_cls,num_cls,save_dir):
    # local_map_all_tensor: (B,N,Cls)
    # global_map_all_tensor: (B,N,1)
    # y_true_array: (B,1)
    # original_imgs_array: (B,W,H,Channel)
    # original_targets_array: (B,1)


    for cls_id in range(num_cls):
        select_id = np.where(np.array(original_targets_array.squeeze(1)) == cls_id)[0].tolist()[:sampling_num_per_cls]
        for sample_id in select_id:
            original_img_array = original_imgs_array[sample_id] # (W,H,Channel)
            w,h,c = original_img_array.shape
            check_gt = y_true_array[sample_id][0]  # int

            if global_map_all_tensor is not None:
                global_map_tensor = global_map_all_tensor[sample_id] # (N,1)
                global_map_tensor_resize = F.interpolate(global_map_tensor.view(1,1,map_size,map_size), size=w, mode='bilinear') # (1,1,w,h)
                global_map_array_resize = global_map_tensor_resize.numpy()

                save_global_att_imgname = save_dir + 'cls{}_sample{}_checkid{}_global_att.png'.format(cls_id, sample_id,
                                                                                                      check_gt)
                save_img_att(original_img_array, global_map_array_resize[0], save_global_att_imgname)

            local_map_tensor = local_map_all_tensor[sample_id] # (N,Cls)
            local_map_tensor_resize = F.interpolate(local_map_tensor.permute(1, 0).view(1, num_cls, map_size, map_size), size=w,
                                                     mode='bilinear') # (1,Cls,w,h)
            local_map_array_resize = local_map_tensor_resize.numpy()

            save_local_att_imgname = save_dir + 'cls{}_sample{}_checkid{}_local_att.png'.format(cls_id, sample_id,
                                                                                                  check_gt)
            save_img_att(original_img_array, local_map_array_resize[0],save_local_att_imgname)




def MedMNIST_TEST_Tasks(Test_Loaders, model, save_dir, phase):
    data_flag = 'organcmnist'
    model.eval()
    params_num = count_parameters(model)

    y_true_0 = torch.tensor([],dtype=torch.long)
    y_score_0 = torch.tensor([])

    main_outch = Test_Loaders.dataset.num_cls

    with torch.no_grad():
        for i, (data, targets) in enumerate(Test_Loaders):
            # net.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'net_%03d_%03d.pkl' % (epoch-1,1))))
            # if 'mnist' in mode:
            #     data = data.repeat(1,3,1,1)
            data = data.cuda()

            logits = model(data)

            y_true_0 = torch.cat((y_true_0, targets), 0)

            if Test_Loaders.dataset.dataset_name == 'ChestMNIST':
                sigm = nn.Sigmoid()
                task_predict = sigm(logits.cpu())
            else:
                if main_outch == 2:
                    sigm = nn.Sigmoid()
                    task_predict = sigm(logits.cpu().squeeze(1))
                else:
                    task_predict = logits.cpu()
                    task_predict = task_predict.softmax(dim=-1)

            y_score_0 = torch.cat((y_score_0, task_predict), 0)


    y_score_0 = y_score_0.detach().numpy()
    y_true_0 = y_true_0.numpy()

    if Test_Loaders.dataset.dataset_name == 'ChestMNIST':
        auc_0 = getAUC(y_true_0, y_score_0, 'multi-label-binary-class')
        acc_0 = getACC(y_true_0, y_score_0, 'multi-label-binary-class')
    else:
        if main_outch == 2:
            auc_0 = getAUC(y_true_0, y_score_0, 'binary')
            acc_0 = getACC(y_true_0, y_score_0, 'binary')
        else:
            auc_0 = getAUC(y_true_0, y_score_0, 'multi_cls')
            acc_0 = getACC(y_true_0, y_score_0, 'multi_cls')

    # save
    np.save(save_dir+'predicted_prob.npy',y_score_0)
    np.save(save_dir + 'gt.npy', y_true_0)

    with open("%s/testout_index.txt" % (save_dir), "a") as f:
        f.writelines(
            ["\n", "params: ", str(params_num),"\n",
             phase, ":", '\n', "Accuracy_0:", str(acc_0), ",auc_0: ", str(auc_0), '\n'
             ])

    # if phase == 'test':
    #     test_heatmap(Test_Loaders, model, save_dir)

    return acc_0, auc_0



def getAUC(y_true, y_score, task):
    """AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label-binary-class":
        auc = 0
        for i in range(y_score.shape[1]):
            label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
            auc += label_auc
        ret = auc / y_score.shape[1]
    elif task == "binary":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        ret = auc / y_score.shape[1]

    return ret


def getACC(y_true, y_score, task, threshold=0.5):
    """Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label-binary-class":
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == "binary":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = accuracy_score(y_true, y_score > threshold)
    else:
        ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

    return ret


def compute_psnr_ssim_AME(tensor_out, tensor_gt):
    n = tensor_out.size()[0]
    tensor_out = tensor_out * 0.5 + 0.5
    tensor_gt = tensor_gt * 0.5 + 0.5

    numpy2 = tensor_out.clamp(min=0.0, max=1.0).permute(0, 2, 3, 1).numpy()
    # numpy1 = np.clip((numpy1*std+mean)*255,0,255)

    numpy1 = tensor_gt.permute(0, 2, 3, 1).numpy()
    # numpy2 = np.clip((numpy2 * std + mean) * 255, 0, 255)

    psnr_list = []
    ssim_list = []
    ame_list = []

    for i in range(n):
        data_range = 1  # max(numpy1.max(), numpy2.max()) - min(numpy1.min(), numpy2.min())

        psnr = peak_signal_noise_ratio(numpy1[i], numpy2[i], data_range=data_range)
        ssim = structural_similarity(numpy1[i], numpy2[i], multichannel=True, data_range=data_range, channel_axis=-1)
        ame = np.mean(np.abs(tensor_out[i].numpy() - tensor_gt[i].numpy()))

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        ame_list.append(ame)

    return psnr_list, ssim_list, ame_list