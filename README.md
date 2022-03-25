# Fast Differentiable Matrix Sqrt Root

<div align=center><center><b>Geometric Interpretation of Matrix Square Root and Inverse Square Root</b></center></div>
<div align=center><img src="MatSqrt_Cover.jpg" width="60%"/></div>

This repository constains the official Pytorch implementation of ICLR 22 paper ["Fast Differentiable Matrix Square Root"](https://arxiv.org/pdf/2201.08663.pdf) and the expanded journal version ["Fast Differentiable Matrix Square Root and Inverse Square Root"](https://arxiv.org/pdf/2201.12543.pdf).

You can find the presentation of our work by the [slides](https://docs.google.com/presentation/d/1JsQMANEQNPG2aP7MOdbx4USHJS-MtzTTsx0t0rEtSXc/edit) and [poster](https://github.com/KingJamesSong/FastDifferentiableMatSqrt/blob/main/iclr22_sqrt.png).

## Usages

Check [torch_utils.py](https://github.com/KingJamesSong/FastDifferentiableMatSqrt/blob/main/torch_utils.py) for the implementation.
Minimal exemplery usage is given as follows:

```python
# Import and define function
from torch_utils import *
FastMatSqrt=MPA_Lya.apply
FastInvSqrt=MPA_Lya_Inv.apply

# For any batched matrices, compute their square root or inverse square root:
rand_matrix = torch.randn(5,32,32)
rand_cov = rand_matrix.bmm(rand_matrix.transpose(1,2))
rand_cov_sqrt = FastMatSqrt(rand_cov)
rand_inv_sqrt = FastInvSqrt(rand_cov)
```

## Computer Vision Experiments

All the codes for the following experiments are available: 
- [Decorrelated Batch Normalization (BN)](https://github.com/KingJamesSong/FastDifferentiableMatSqrt/tree/main/Decorrelated%20BN)
- [Second-order Vision Transformer (So-ViT)](https://github.com/KingJamesSong/FastDifferentiableMatSqrt/tree/main/So-ViT)
- [Neural Style Transfer by Whitening and Coloring Transform (WCT)](https://github.com/KingJamesSong/FastDifferentiableMatSqrt/tree/main/Neural%20Style%20Transfer)
- [Temporal-Attentive Covariance Pooling (TACP) for Video Action Recognition](https://github.com/KingJamesSong/FastDifferentiableMatSqrt/tree/main/TACP)
- [Global Covariance Pooling for Large-Scale and Fine-grained Visual Recognition](https://github.com/KingJamesSong/DifferentiableSVD)


## Citation

Please consider citing our paper if you think the code is helpful to your research.

```
@inproceedings{song2022fast,
  title={Fast Differentiable Matrix Square Root},
  author={Song, Yue and Sebe, Nicu and Wang, Wei},
  booktitle={ICLR},
  year={2022}
}
```

```
@article{song2022fast2,
  title={Fast Differentiable Matrix Square Root and Inverse Square Root},
  author={Song, Yue and Sebe, Nicu and Wang, Wei},
  journal={arXiv preprint arXiv:2201.12543},
  year={2022}
}
```

## Contact

If you have any questions or suggestions, please feel free to contact me

`yue.song@unitn.it`
