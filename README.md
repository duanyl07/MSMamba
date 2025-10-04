# MSMamba

A New Multiscale Superpixel Mamba for Hyperspectral Image Classification

has been accepted by IEEE Transactions on Geoscience and Remote Sensing (TGRS)

Abstract: State-space models (SSMs), particularly the Mamba model, have recently garnered significant attention in hyperspectral image (HSI) classification tasks due to their capability to model global dependencies with linear computational complexity. However, existing SSM-based approaches predominantly adopt patch-based strategies with single-scale information, which inherently limits their ability to effectively characterize boundary structures and shapes in complex land-cover regions. We propose a novel multiscale superpixel Mamba (MSMamba) framework for HSI classification that captures long-range dependencies using a two-stage multiscale Mamba (MMamba) structure. In the first stage, a multiscale superpixel generation and fusion (MSSGF) strategy and an MMamba module are introduced. Specifically, multiscale superpixels are fused and unfolded into bidirectional sequences. The MMamba module models global dependencies through cross-scale sequence unfolding. In the second stage, we propose a multishape fusion mechanism to integrate the global features of MMamba under varying shape configurations, aiming to reinforce boundary-sensitive representation learning. In addition, our MSMamba introduces a dual-branch architecture that synergistically integrates MMamba with a multiscale spectral–spatial convolutional network (SSCNN) to enhance finegrained local spectral–spatial details. Extensive experiments conducted on four publicly available HSI datasets demonstrate that our MSMamba method significantly outperforms related methods in classification performance, particularly excelling at identifying complex terrain boundarie.

Environment: Mamba

Run  main.py


Early Access is available now at [paper](https://ieeexplore.ieee.org/abstract/document/11175220)

If this work is helpful to you, please citing our work as follows: 

Y. Duan, L. Yu, J. Chen, Z. Zeng, J. Li and A. Plaza, "A New Multiscale Superpixel Mamba for Hyperspectral Image Classification," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2025.3612566.


Or

@ARTICLE{MSMamba,
  author={Duan, Yilin and Yu, Long and Chen, Jia and Zeng, Zhaozhao and Li, Jun and Plaza, Antonio},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A New Multiscale Superpixel Mamba for Hyperspectral Image Classification}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Computational modeling;Convolutional neural networks;Transformers;Three-dimensional displays;Solid modeling;Hyperspectral imaging;Geology;Shape;Semantics;Hyperspectral image classification;state-space models;convolutional neural networks},
  doi={10.1109/TGRS.2025.3612566}}

