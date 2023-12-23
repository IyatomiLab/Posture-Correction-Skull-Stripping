# PCSS
![image](https://github.com/IyatomiLab/Posture-Correction-Skull-Stripping/assets/64403395/c27cdc92-e40c-43d4-9508-2e7a87558ab3)

[![IEEE Xplore](https://img.shields.io/badge/Accepted-IEEE%20Access-%2300629B%09)](https://ieeexplore.ieee.org/abstract/document/10288431)

**PCSS: Skull Stripping With Posture Correction From 3D Brain MRI for Diverse Imaging Environment**<br>
**Author**: Kei Nishimaki, Kumpei Ikuta, Shingo Fujiyama, [Kenichi Oishi](https://www.hopkinsmedicine.org/profiles/details/kenichi-oishi), [Hitoshi Iyatomi](https://iyatomi-lab.info/english-top).<br>

Department of Applied Informatics, Graduate School of Science and Engineering, Hosei University, Tokyo, Japan <br>
The Russell H. Morgan Department of Radiology and Radiological Science, The Johns Hopkins University School of Medicine, Baltimore, MD, USA <br>

**Abstract**: *A subject’s head position in magnetic resonance imaging (MRI) scanners can vary significantly with the imaging environment and disease status. This variation is known to influence the accuracy of skull stripping (SS), a method to extract the brain region from the whole head image, which is an essential initial step to attain high performance in various neuroimaging applications. However, existing SS methods have failed to accommodate this wide range of variation. To achieve accurate, consistent, and fast SS, we introduce a novel two-stage methodology that we call posture correction skull stripping (PCSS): the first involves adjusting the subject’s head angle and position, and the second involves the actual SS to generate the brain mask. PCSS also incorporates various machine learning techniques, such as a weighted loss function, adversarial training from generative adversarial networks, and ensemble methods. Thorough evaluations conducted on five publicly accessible datasets show that the PCSS method outperforms current state-of-the-art techniques in SS performance, achieving an average increase of 1.38 points on the Dice score and demonstrating the contributions of each PCSS component technique.*

Paper: https://ieeexplore.ieee.org/abstract/document/10288431<br>
Submitted for publication in the **IEEE Access**<br>

We recommend that you also check out the following studies related to ours.<br>
OpenMAP-T1 provides a more robust skull-stripping mask and 280 anatomical parcellation map.<br>
Nishimaki et al. "[OpenMAP-T1: A Rapid Deep-Learning Approach to Parcellate 280 Anatomical Regions to Cover the Whole Brain
Author](https://github.com/OishiLab/OpenMAP-T1)"


## Installation Instructions
0. install python and make virtual environment<br>
python3.8 or later is recommended.

1. Clone this repository:
```
git clone https://github.com/IyatomiLab/Posture-Correction-Skull-Stripping.git
```
2. Please install PyTorch compatible with your environment.<br>
https://pytorch.org/

Once you select your environment, the required commands will be displayed.

<img width="485" alt="image" src="https://github.com/OishiLab/OpenMAP-T1-V1/assets/64403395/eb092ff6-6597-4237-ac3a-aa0695bff631">

If you want to install an older Pytorch environment, you can download it from the link below.<br>
https://pytorch.org/get-started/previous-versions/

4. Go into the repository and install:
```
cd Posture-Correction-Skull-Stripping
pip install -r requirements.txt
```

## How to use it
Using PCSS is straightforward. You can use it in any terminal on your linux system. The PCSS command was installed automatically. We provide CPU as well as GPU support. Running on GPU is a lot faster though and should always be preferred. Here is a minimalistic example of how you can use PCSS.
```
python3 parcellation.py -i INPUR_DIRNAME -o OUTPUT_DIRNAME -m MODEL_DIRNAME
```
If you want to specify the GPU, please add ```--gpu```.
```
python3 parcellation.py -i INPUR_DIRNAME -o OUTPUT_DIRNAME -m MODEL_DIRNAME --gpu 1
```

### Folder
All images you input must be in NifTi format and have a .nii extension.
```
INPUR_DIRNAME/
  ├ A.nii
  ├ B.nii
  ├ *.nii

OUTPUT_DIRNAME/
  ├ A.nii
  ├ B.nii
  ├ *.nii

MODEL_DIRNAME/
  ├ PENet.pth
  └ SSNet.pth
```
## How to download the pretrained model.
You can get the pretrained model from the this link.
[Link of pretrained model](https://drive.google.com/drive/folders/1FIdfFGf3FJ3CR1pMTYmd47IW_gMZox42?usp=sharing)

## FAQ
* **How much GPU memory do I need to run PCSS?** <br>
We ran all our experiments on NVIDIA RTX3090 GPUs with 24 GB memory. For inference you will need less, but since inference in implemented by exploiting the fully convolutional nature of CNNs the amount of memory required depends on your image. Typical image should run with less than 4 GB of GPU memory consumption. If you run into out of memory problems please check the following: 1) Make sure the voxel spacing of your data is correct and 2) Ensure your MRI image only contains the head region.

* **Will you provide the training code as well?** <br>
No. The training code is tightly wound around the data which we cannot make public.


## Citation
```
@article{nishimaki2023pcss,
  title={PCSS: Skull Stripping with Posture Correction from 3D Brain MRI for Diverse Imaging Environment},
  author={Nishimaki, Kei and Ikuta, Kumpei and Fujiyama, Shingo and Oishi, Kenichi and Iyatomi, Hitoshi},
  journal={IEEE Access},
  year={2023},
  publisher={IEEE}
}
```
