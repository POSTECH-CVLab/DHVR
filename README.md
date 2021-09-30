# Deep Hough Voting for Robust Global Registration, ICCV, 2021
### [Project Page](https://cvlab.postech.ac.kr/research/DHVR/) | [Paper](https://arxiv.org/abs/2109.04310) | [Video](https://youtu.be/lPv5kKQGxZQ)

[Deep Hough Voting for Robust Global Registration](https://cvlab.postech.ac.kr/research/DHVR/)  
 [Junha Lee](https://junha-l.github.io)<sup>1</sup>,
 Seungwook Kim<sup>1</sup>,
 [Minsu Cho](http://tancik.com/)<sup>1</sup>,
 [Jaesik Park](http://jonbarron.info/)<sup>1</sup><br>
 <sup>1</sup>POSTECH CSE & GSAI<br>
in ICCV 2021 

<div style="text-align:center">
<img src="assets/pipeline.png" alt="An Overview of the proposed pipeline"/>
</div>

## Overview

Point cloud registration is the task of estimating the rigid transformation that aligns a pair of point cloud fragments. We present an efficient and robust framework for pairwise registration of real-world 3D scans, leveraging Hough voting in the 6D transformation parameter space. First, deep geometric features are extracted from a point cloud pair to compute putative correspondences. We then construct a set of triplets of correspondences to cast votes on the 6D Hough space, representing the transformation parameters in sparse tensors. Next, a fully convolutional refinement module is applied to refine the noisy votes. Finally, we identify the consensus among the correspondences from the Hough space, which we use to predict our final transformation parameters. Our method outperforms state-of-the-art methods on 3DMatch and 3DLoMatch benchmarks while achieving comparable performance on KITTI odometry dataset. We further demonstrate the generalizability of our approach by setting a new state-of-the-art on ICL-NUIM dataset, where we integrate our module into a multi-way registration pipeline.

## Coming Soon :rocket: :rocket:

## Citing our paper

```
@InProceedings{lee2021deephough, 
    title={Deep Hough Voting for Robust Global Registration},
    author={Junha Lee and Seungwook Kim and Minsu Cho and Jaesik Park},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2021}
}
```