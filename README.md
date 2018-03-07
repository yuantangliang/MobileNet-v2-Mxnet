# MobileNet-V2-Mxnet

### Introduction

This is a Mxnet implementation of Google's MobileNets (v2). For details, please read the following papers:
- [v2] [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)

This Model is converted from [MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe)
### Pretrained Models on ImageNet

We provide pretrained MobileNet models on ImageNet, which achieve slightly better accuracy rates than the original ones reported in the paper. 

The top-1/5 accuracy rates by using single center crop (crop size: 224x224, image size: 256xN):

Network|Top-1|Top-5|sha256sum|Architecture
:---:|:---:|:---:|:---:|:---:
MobileNet v2| 71.90| 90.49| a3124ce7 (13.5 MB)| [netscope](http://ethereon.github.io/netscope/#/gist/d01b5b8783b4582a42fe07bd46243986)


### Evaluate Models with a single image

Evaluate MobileNet v2:

`python mobile_v2_eval.py `

Expected Outputs,the output is different from caffe version, it could be cause by difference in preprocess:

```
(0.38443729, "'n02124075 Egyptian cat'")
(0.11267297, "'n02123159 tiger cat'")
(0.074108832, "'n02123045 tabby, tabby cat'")
(0.058392685, "'n02119022 red fox, Vulpes vulpes'")
(0.031767886, "'n02094258 Norwich terrier'")
```

### Todo List ###
- [X] convert symbol
- add python symbol

