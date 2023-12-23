import argparse
import glob
import os
from functools import partial

import nibabel as nib
import numpy as np
import torch
from nibabel import processing
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)

from utils.afterprocess import (cut_pad_voxel, estimate_neck, normalize,
                                rotate_voxel, strip)
from utils.PENet import PENet
from utils.SSNet import SSNet


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input folder")
    parser.add_argument("-o", help="output folder")
    parser.add_argument("-m", help="model folder")
    parser.add_argument("--gpu", default=0, help="GPU")
    return parser.parse_args()

def load_model(opt, device):
    penet = PENet(2)
    penet.load_state_dict(torch.load(os.path.join(opt.m, "PENet.pth")))
    penet.eval()

    ssnet = SSNet(1)
    ssnet.load_state_dict(torch.load(os.path.join(opt.m, "SSNet.pth")))
    ssnet.to(device)
    ssnet.eval()
    return penet, ssnet

def main():
    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using PCSS:\n"
        "Kei Nishimaki, Kumpei Ikuta, Shingo Fujiyama, Kenichi Oishi, Hitoshi Iyatomi. (2023). "
        "PCSS: Skull Stripping With Posture Correction From 3D Brain MRI for Diverse Imaging Environment. "
        "IEEE Access.\n#######################################################################\n"
        )
    opt = create_parser()
    device = torch.device("cuda", int(opt.gpu)) if torch.cuda.is_available() else "cpu"
    penet, ssnet = load_model(opt, device)
    print("load complete !!")

    pathes = sorted(glob.glob(os.path.join(opt.i, "**/*.nii"), recursive=True))
    for path in tqdm(pathes):
        os.makedirs(opt.o, exist_ok=True)
        odata = nib.squeeze_image(nib.as_closest_canonical(nib.load(path)))
        data = processing.conform(odata, out_shape=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0), order=1)
        voxel = data.get_fdata().astype("float32")
        
        rot, y_0 = estimate_neck(voxel, penet)
        rot_voxel, rot_x = rotate_voxel(voxel, rot, y_0)
        pad_voxel = cut_pad_voxel(rot_voxel, rot_x)
        
        voxel = np.clip(pad_voxel, 0, 4 * np.std(pad_voxel))
        voxel = normalize(voxel) * 255
        voxel = voxel / 127.5 - 1.0

        coronal = voxel.transpose(1, 2, 0)
        sagittal = voxel
        transverse = voxel.transpose(2, 1, 0)

        c_out = strip(coronal, ssnet, device).permute(2,0,1)
        s_out = strip(sagittal, ssnet, device)
        t_out = strip(transverse, ssnet, device).permute(2,1,0)

        e_out = ((c_out + s_out + t_out) / 3) > 0.5
        e_out = e_out.cpu().numpy()
        stripped = pad_voxel * e_out
            
        affine = np.eye(4)
        affine[0][3] = -128
        affine[1][3] = -128
        affine[2][3] = -128
        nii = nib.Nifti1Image(stripped.astype(np.float32), affine=affine)
        nib.save(nii, os.path.join(opt.o, f"{os.path.basename(path)}"))
    return

if __name__ == "__main__":
    main()