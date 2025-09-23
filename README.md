<p align="center">

  <h1 align="center">HeightFormer: Learning Height Prediction in Voxel Features for Roadside Vision Centric 3D Object Detection via Transformer</h1>
  
  </p>

<!--
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-TITS-00629b.svg)](https://ieeexplore.ieee.org/document/11005676)

<p align="center">
<img src="docs/assets/height3d_fig3.png" width="800" alt="" class="img-responsive">
</p>
<p align="center">
<img src="docs/assets/height3d_fig8.png" width="800" alt="" class="img-responsive">
</p>
-->

# Getting Started

- [Installation](docs/install.md)
- [Prepare Dataset](docs/prepare_dataset.md)

Train HeightFormer with 8 GPUs
```
python [EXP_PATH] --amp_backend native -b 8 --gpus 8
```
Eval HeightFormer with 1 GPU
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 1 --gpus 1
```

# Acknowledgment
This project is not possible without the following codebases.
* [Height3D](https://github.com/zhangzhang2024/Height3D) 
* [BEVHeight](https://github.com/ADLab-AutoDrive/BEVHeight)
* [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)

# Citation
If you use HeightFormer in your research, please cite our work by using the following BibTeX entry:
```
@article{zhang2025heightformer,
  title={Heightformer: Learning height prediction in voxel features for roadside vision centric 3d object detection via transformer},
  author={Zhang, Zhang and Sun, Chao and Yue, Chao and Wen, Da and Chen, Yujie and Wang, Tianze and Leng, Jianghao},
  journal={arXiv preprint arXiv:2503.10777},
  year={2025}
}
```
