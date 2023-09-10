# Single Image 3D Object Estimation with Primitive Graph Networks


The official code for "[Single Image 3D Object Estimation with Primitive Graph Networks](https://arxiv.org/abs/2109.04153)" (ACM Multimedia 2021).



https://github.com/hailieqh/3D-Object-Primitive-Graph/assets/25678575/04b4b739-1e2f-404d-beed-6a5619ea34d7



## Prerequisites
- Python 3.6
- Pytorch 1.0.0
- Cuda version 9.0


## Installation
```
pip install -r requirements.txt
```


## Data Preparation

Our primitive annotations can be downloaded from [primitive_resources](https://drive.google.com/file/d/1wXSudsd3Am86ZuPRJV1h4oApTIIFecsg/view?usp=drive_link).

We provide Chair from [Pix3D](https://github.com/xingyuansun/pix3d) as an example.

Please refer to process_data/all/steps.txt for the detailed data processing pipeline if you are insterested in applying our method to your data.


## Training and testing
We provide the training and testing steps of Chair in scritps.sh.


## Visualization
For visualization of the results, please refer to process_data/all/code/visualization/visualize_everything.m.


## Acknowledgement

We thank [Pix3D](http://pix3d.csail.mit.edu/papers/pix3d_cvpr.pdf), [3D-PRNN](http://dhoiem.cs.illinois.edu/publications/3D-PRNN_ICCV17_Zou.pdf), [PQ-Net](https://arxiv.org/pdf/1911.10949.pdf), [Tulsiani et al.](https://arxiv.org/pdf/1612.00404.pdf) and other primitive estimation works for their wonderful contributions.


## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{he2021single,
  title={Single Image 3D Object Estimation with Primitive Graph Networks},
  author={He, Qian and Zhou, Desen and Wan, Bo and He, Xuming},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={2353--2361},
  year={2021}
}
```
