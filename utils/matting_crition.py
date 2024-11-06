import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_gradient_penalty( sample_map ,preds, targets):

    #sample_map for unknown area
    scale = sample_map.shape[0]*262144/torch.sum(sample_map)

    #gradient in x
    sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(dtype=preds.type())
    delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
    delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)

    #gradient in y 
    sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(dtype=preds.type())
    delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
    delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)

    #loss
    loss = (F.l1_loss(delta_pred_x*sample_map, delta_gt_x*sample_map)* scale + \
        F.l1_loss(delta_pred_y*sample_map, delta_gt_y*sample_map)* scale + \
        0.01 * torch.mean(torch.abs(delta_pred_x*sample_map))* scale +  \
        0.01 * torch.mean(torch.abs(delta_pred_y*sample_map))* scale)

    return loss



def unknown_l1_loss( sample_map, preds, targets):
    
    scale = sample_map.shape[0]*262144/torch.sum(sample_map)
    # scale = 1

    loss = F.l1_loss(preds*sample_map, targets*sample_map)*scale
    return loss
