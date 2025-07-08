<p align="center">

  <h1 align="center">Height3D: A Roadside Visual Framework Based on Height Prediction in Real 3-D Space</h1>
  
  </p>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-TITS-00629b.svg)](https://ieeexplore.ieee.org/document/11005676)

<p align="center">
<img src="docs/assets/height3d_fig3.png" width="800" alt="" class="img-responsive">
</p>
<p align="center">
<img src="docs/assets/height3d_fig8.png" width="800" alt="" class="img-responsive">
</p>

# Getting Started

- [Installation](docs/install.md)
- [Prepare Dataset](docs/prepare_dataset.md)

Train Height3D with 8 GPUs
```
python [EXP_PATH] --amp_backend native -b 8 --gpus 8
```
Eval Height3D with 1 GPU
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 1 --gpus 1
```

# Acknowledgment
This project is not possible without the following codebases.
* [BEVHeight](https://github.com/ADLab-AutoDrive/BEVHeight)
* [DAIR-V2X](https://github.com/AIR-THU/DAIR-V2X)

# Citation
If you use Height3D in your research, please cite our work by using the following BibTeX entry:
```
@article{zhang2025height3d,
  title={Height3d: A roadside visual framework based on height prediction in real 3-d space},
  author={Zhang, Zhang and Sun, Chao and Wang, Bo and Guo, Bin and Wen, Da and Zhu, Tianyi and Ning, Qili},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2025},
  publisher={IEEE}
}
```
