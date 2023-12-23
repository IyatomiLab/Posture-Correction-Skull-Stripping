# PCSS
![image](https://github.com/IyatomiLab/Posture-Correction-Skull-Stripping/assets/64403395/c27cdc92-e40c-43d4-9508-2e7a87558ab3)

[![IEEE Xplore](https://img.shields.io/badge/Accepted-IEEE%20Access-%2300629B%09)](https://ieeexplore.ieee.org/abstract/document/10288431)

**OpenMAP-T1: A Rapid Deep-Learning Approach to Parcellate 280 Anatomical Regions to Cover the Whole Brain**<br>
**Author**: Kei Nishimaki, Kumpei Ikuta, [Kenichi Oishi](https://www.hopkinsmedicine.org/profiles/details/kenichi-oishi), [Hitoshi Iyatomi](https://iyatomi-lab.info/english-top).<br>

Department of Applied Informatics, Graduate School of Science and Engineering, Hosei University, Tokyo, Japan <br>
The Russell H. Morgan Department of Radiology and Radiological Science, The Johns Hopkins University School of Medicine, Baltimore, MD, USA <br>

**Abstract**: *A subject’s head position in magnetic resonance imaging (MRI) scanners can vary significantly with the imaging environment and disease status. This variation is known to influence the accuracy of skull stripping (SS), a method to extract the brain region from the whole head image, which is an essential initial step to attain high performance in various neuroimaging applications. However, existing SS methods have failed to accommodate this wide range of variation. To achieve accurate, consistent, and fast SS, we introduce a novel two-stage methodology that we call posture correction skull stripping (PCSS): the first involves adjusting the subject’s head angle and position, and the second involves the actual SS to generate the brain mask. PCSS also incorporates various machine learning techniques, such as a weighted loss function, adversarial training from generative adversarial networks, and ensemble methods. Thorough evaluations conducted on five publicly accessible datasets show that the PCSS method outperforms current state-of-the-art techniques in SS performance, achieving an average increase of 1.38 points on the Dice score and demonstrating the contributions of each PCSS component technique.*

Paper: [Not yet](https://ieeexplore.ieee.org/abstract/document/10288431)<br>
Submitted for publication in the **IEEE Access**<br>

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
