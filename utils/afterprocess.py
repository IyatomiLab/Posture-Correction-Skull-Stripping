import math
from typing import Tuple

import cv2
import numpy as np
import torch
from scipy import ndimage
from utils.PENet import PENet


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def serach_slice(voxel: np.ndarray) -> np.ndarray:
    count_list = np.array([np.count_nonzero(s) for s in voxel[88:168]])
    slice_num = int(np.median(np.where(count_list == count_list.max())[0]))
    return voxel[slice_num+88]

def estimate_neck(voxel: np.ndarray, penet: PENet) -> Tuple[float, float]:
    original = serach_slice(voxel)
    original = np.clip(original, 0, 4*np.std(voxel))
    image = cv2.resize(original, (128, 128), interpolation=cv2.INTER_LINEAR)
    image = np.clip(image, 0, image.max())
    image = normalize(image)
    image = torch.tensor(image.reshape(1, 1, 128, 128), requires_grad=False)
    
    penet.eval()
    with torch.inference_mode():
        py = penet(image)

    pa = py[0][0].numpy()
    pb = py[0][1].numpy()
    y_0 = pa * 0 + pb        
    rot = -1 * (180 * np.arctan(pa)) / np.pi
    return rot, y_0

def rotate_point(x: float, y: float, angle_deg: float) -> Tuple[float, float]:
    angle_rad = np.deg2rad(angle_deg)
    cos = math.cos(angle_rad)
    sin = math.sin(angle_rad)
    rx = 128 + cos * (x - 128) - sin * -(y - 128)
    ry = 128 + sin * (x - 128) + cos * -(y - 128)
    return rx, 256 - ry

def rotate_voxel(voxel: np.ndarray, rot: float, y_0: float) -> Tuple[np.ndarray, float]:
    rot_voxel = ndimage.rotate(voxel, rot, mode='nearest', reshape=False, order=1, axes=(1,2))
    rot_x = rotate_point(y_0, 0, rot)[0]
    return rot_voxel, rot_x

def cut_pad_voxel(rot_voxel: np.ndarray, rot_x: float) -> np.ndarray:
    position = rot_x
    threshold = 60
    if position > threshold:
        base = int(position - threshold)
        pad_voxel = rot_voxel[:,:,base:]
        pad_voxel = np.pad(pad_voxel, [(0, 0), (0, 0), (0, base)], "constant")
    else:
        base = int(threshold - position)
        pad_voxel = rot_voxel[:,:,:256-base]
        pad_voxel = np.pad(pad_voxel, [(0, 0), (0, 0), (base, 0)], "constant")
    return pad_voxel

def strip(voxel, model, device):
    model.eval()
    with torch.inference_mode():
        output = torch.zeros(256, 256, 256).to(device)
        for i, v in enumerate(voxel):
            image = v.reshape(1,1,256,256)
            image = torch.tensor(image).to(device)
            x_out = torch.sigmoid(model(image)).detach()
            if i == 0:
                output[0] = x_out
            else:
                output[i] = x_out
        return output.reshape(256, 256, 256)