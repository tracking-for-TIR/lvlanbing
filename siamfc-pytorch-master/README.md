# SiamFC - PyTorch

> Highlights of this update:
> - Higher scores with more stable training performance.
> - Faster training (~11 minutes to train one epoch on GOT-10k on a single GPU).
> - Added MIT LICENSE.
> - Organized code.
> - Uploaded pretrained weights. ([Google Drive](https://drive.google.com/file/d/1UdxuBQ1qtisoWYFZxLgMFJ9mJtGVw6n4/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1MTVXylPrSqpqmVD4iBwbpg) (password: wbek))

A clean PyTorch implementation of SiamFC tracker described in paper [Fully-Convolutional Siamese Networks for Object Tracking](https://www.robots.ox.ac.uk/~luca/siamese-fc.html). The code is evaluated on 7 tracking datasets ([OTB (2013/2015)](http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html), [VOT (2018)](http://votchallenge.net), [DTB70](https://github.com/flyers/drone-tracking), [TColor128](http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html), [NfS](http://ci2cv.net/nfs/index.html) and [UAV123](https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx)), using the [GOT-10k toolkit](https://github.com/got-10k/toolkit).

## Installation

Install Anaconda, then install dependencies:

```bash
# install PyTorch >= 1.0
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
# intall OpenCV using menpo channel (otherwise the read data could be inaccurate)
conda install -c menpo opencv
# install GOT-10k toolkit
pip install got10k
```

[GOT-10k toolkit](https://github.com/got-10k/toolkit) is a visual tracking toolkit that implements evaluation metrics and tracking pipelines for 9 popular tracking datasets.

## Training the tracker

1. Setup the training dataset in `tools/train.py`. Default is the GOT-10k dataset located at `~/data/GOT-10k`.

2. Run:

```
python tools/train.py
```

## Evaluate the tracker

1. Setup the tracking dataset in `tools/test.py`. Default is the OTB dataset located at `~/data/OTB`.

2. Setup the checkpoint path of your pretrained model. Default is `pretrained/siamfc_alexnet_e50.pth`.

3. Run:

```
python tools/test.py
```

## Running the demo

1. Setup the sequence path in `tools/demo.py`. Default is `~/data/OTB/Crossing`.

2. Setup the checkpoint path of your pretrained model. Default is `pretrained/siamfc_alexnet_e50.pth`.

3. Run:

```
python tools/demo.py
```
