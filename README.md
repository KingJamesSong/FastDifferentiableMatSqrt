# Fast Differentiable Matrix Sqrt Root

This repository constains the official Pytorch implementation of ICLR 22 paper "Fast Differentiable Matrix Square Root".

## Experiments

All the codes for the experiments are available, including decorrelated batch normalization and second-order vision transformer.

## Usages

Check [torch_utils.py](https://github.com/KingJamesSong/FastDifferentiableMatSqrt/blob/main/torch_utils.py) for the implementation.
Minimal exemplery usage is given as follows:

`from torch_utils import *`
`FastMatSqrt=MPA_Lya.apply`

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

## Contact

If you have any questions or suggestions, please feel free to contact me

`yue.song@unitn.it`
