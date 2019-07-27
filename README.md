此库克隆于[PSENet](https://github.com/whai362/PSENet)<br><br>
分割任务常见的问题是，图像中目标区域面积很小，即正例少负样本多，正负样本极其不均衡。此类问题在医疗领域也大量存在，并有大量损失函数被发明:[从loss处理图像分割中类别极度不均衡的状况---keras](https://blog.csdn.net/m0_37477175/article/details/83004746)<br><br>
PSENet中使用dice loss作为损失函数。<br>
dice收集的一些讲解(目前没找到靠谱的)：[dice就是真实结果与预测结果的相交面积/全面积](https://blog.csdn.net/legend_hua/article/details/80610790),与IOU交并比极其相似，其公式为[2|AnB|(|A|+|B|)](https://zhuanlan.zhihu.com/p/36664544)<br>
[dice loss(csdn)](https://blog.csdn.net/qq_34106574/article/details/89916947)的总结中也发现了dice可能效果并不是很好，可以用bce代替
[dice vs bce](https://stats.stackexchange.com/questions/321460/dice-coefficient-loss-function-vs-cross-entropy)


# Shape Robust Text Detection with Progressive Scale Expansion Network


## Requirements
* Python 2.7
* PyTorch v0.4.1+
* pyclipper
* Polygon2
* OpenCV 3.4 (for c++ version pse)
* opencv-python 3.4

## Introduction
Progressive Scale Expansion Network (PSENet) is a text detector which is able to well detect the arbitrary-shape text in natural scene.

## Training
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ic15.py
```

## Testing
```
CUDA_VISIBLE_DEVICES=0 python test_ic15.py --scale 1 --resume [path of model]
```

## Eval script for ICDAR 2015 and SCUT-CTW1500
```
cd eval
sh eval_ic15.sh
sh eval_ctw1500.sh
```


## Performance (new version paper)
### [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4&com=evaluation&task=1)
| Method | Extra Data | Precision (%) | Recall (%) | F-measure (%) | FPS (1080Ti) | Model |
| - | - | - | - | - | - | - |
| PSENet-1s (ResNet50) | - | 81.49 | 79.68 | 80.57 | 1.6 | [baiduyun](https://pan.baidu.com/s/17FssfXd-hjsU5i2GGrKD-g)(extract code: rxti); [OneDrive](https://1drv.ms/u/s!Ai5Ldd26Lrzkkx3E1OTZzcNlMz5T) |
| PSENet-1s (ResNet50) | pretrain on IC17 MLT | 86.92 | 84.5 | 85.69 | 1.6 | [baiduyun](https://pan.baidu.com/s/1oKVxHKuT3hdzDUmksbcgAQ)(extract code: aieo); [OneDrive](https://1drv.ms/u/s!Ai5Ldd26Lrzkkx44xvpay4rbV4nW) |
| PSENet-4s (ResNet50) | pretrain on IC17 MLT | 86.1 | 83.77 | 84.92 | 3.8 | [baiduyun](https://pan.baidu.com/s/1oKVxHKuT3hdzDUmksbcgAQ)(extract code: aieo); [OneDrive](https://1drv.ms/u/s!Ai5Ldd26Lrzkkx44xvpay4rbV4nW) |

### [SCUT-CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)
| Method | Extra Data | Precision (%) | Recall (%) | F-measure (%) | FPS (1080Ti) | Model |
| - | - | - | - | - | - | - |
| PSENet-1s (ResNet50) | - | 80.57 | 75.55 | 78.0 | 3.9 | [baiduyun](https://pan.baidu.com/s/1BqJspFwBmHjoqlE0jOrJQg)(extract code: ksv7); [OneDrive](https://1drv.ms/u/s!Ai5Ldd26LrzkkxtlTb-yqBPd1PCn) |
| PSENet-1s (ResNet50) | pretrain on IC17 MLT | 84.84| 79.73 | 82.2 | 3.9 | [baiduyun](https://pan.baidu.com/s/1zonNEABLk4ifseeJtQeS4w)(extract code: z7ac); [OneDrive](https://1drv.ms/u/s!Ai5Ldd26LrzkkxxJcfU1a__6nJTT) |
| PSENet-4s (ResNet50) | pretrain on IC17 MLT | 82.09 | 77.84 | 79.9 | 8.4 | [baiduyun](https://pan.baidu.com/s/1zonNEABLk4ifseeJtQeS4w)(extract code: z7ac); [OneDrive](https://1drv.ms/u/s!Ai5Ldd26LrzkkxxJcfU1a__6nJTT) |

## Performance (old version paper)
### [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4&com=evaluation&task=1) (training with ICDAR 2017 MLT)
| Method | Precision (%) | Recall (%) | F-measure (%) | 
| - | - | - | - |
| PSENet-4s (ResNet152) | 87.98 | 83.87 | 85.88 |
| PSENet-2s (ResNet152) | 89.30 | 85.22 | 87.21 |
| PSENet-1s (ResNet152) | 88.71 | 85.51 | 87.08 |

### [ICDAR 2017 MLT](http://rrc.cvc.uab.es/?ch=8&com=evaluation&task=1)
| Method | Precision (%) | Recall (%) | F-measure (%) | 
| - | - | - | - |
| PSENet-4s (ResNet152) | 75.98 | 67.56 | 71.52 |
| PSENet-2s (ResNet152) | 76.97 | 68.35 | 72.40 |
| PSENet-1s (ResNet152) | 77.01 | 68.40 | 72.45 |

### [SCUT-CTW1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)
| Method | Precision (%) | Recall (%) | F-measure (%) | 
| - | - | - | - |
| PSENet-4s (ResNet152) | 80.49 | 78.13 | 79.29 |
| PSENet-2s (ResNet152) | 81.95 | 79.30 | 80.60 |
| PSENet-1s (ResNet152) | 82.50 | 79.89 | 81.17 |

### [ICPR MTWI 2018 Challenge 2](https://tianchi.aliyun.com/competition/rankingList.htm?spm=5176.100067.5678.4.65166a80jnPm5W&raceId=231651)
| Method | Precision (%) | Recall (%) | F-measure (%) | 
| - | - | - | - |
| PSENet-1s (ResNet152) | 78.5 | 72.1 | 75.2 |

## Results
<div align="center">
  <img src="https://github.com/whai362/PSENet/blob/master/figure/res0.png">
</div>
<p align="center">
  Figure 3: The results on ICDAR 2015, ICDAR 2017 MLT and SCUT-CTW1500
</p>

## Paper Link
[new version paper] [https://arxiv.org/abs/1903.12473](https://arxiv.org/abs/1903.12473)

[old version paper] [https://arxiv.org/abs/1806.02559](https://arxiv.org/abs/1806.02559)

## Other Implements
[tensorflow version (thanks @[liuheng92](https://github.com/liuheng92))] [https://github.com/liuheng92/tensorflow_PSENet](https://github.com/liuheng92/tensorflow_PSENet)
