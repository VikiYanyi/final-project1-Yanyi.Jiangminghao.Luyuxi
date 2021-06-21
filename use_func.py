import sys
import random
import numpy as np
import torch

def transform_input_label(inputs, labels, mode = 'default'):
    
    inputs_ori = inputs.clone()
    labels_new = []
    
    x_center, y_center = np.random.randint(32, size=2)
    square_offset = 8
    
    x_left = max(x_center - square_offset, 0)
    x_right = min(x_center + square_offset, 32)
    y_top = max(y_center - square_offset, 0)
    y_bottom = min(y_center + square_offset, 32)
    
    weight = np.random.beta(1.0, 1.0)
    
    if mode != 'default':
        for i in range(inputs.shape[0]):
            k = np.random.randint(inputs.shape[0])
            
            if mode == 'cutout':
                inputs[i][:, x_left : x_right, y_top : y_bottom] = 0
            
            if mode == 'mixup':
                inputs[i] = inputs_ori[i] * weight + inputs_ori[k] * (1 - weight)
                labels_new.append(labels[k].item())
                
            if mode == 'cutmix':
                weight = 1 - (y_bottom - y_top) * (x_right - x_left) / (32 * 32)
                inputs[i][:, x_left : x_right, y_top : y_bottom] = inputs_ori[k][:, x_left : x_right, y_top : y_bottom]
                labels_new.append(labels[k].item())
                
    labels_new = torch.LongTensor(labels_new)

    return inputs, labels, labels_new, weight