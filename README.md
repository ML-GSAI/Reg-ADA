# DEEP GENERATIVE MODELING ON LIMITED DATA WITH REGULARIZATION BY NONTRANSFERABLE PRE-TRAINED MODELS (ICLR 2023)

This repository provides the official PyTorch implementation of **Reg-ADA** for the following paper (The **Reg-ADA-APA** implementaton is available at https://github.com/ML-GSAI/Reg-ADA-APA):

**DEEP GENERATIVE MODELING ON LIMITED DATA WITH REGULARIZATION BY NONTRANSFERABLE PRE-TRAINED MODELS**
<br>
Yong Zhong, Hongtao Liu, Xiaodong Liu, Fan Bao, Weiran Shen,Chongxuan Li (https://arxiv.org/abs/2208.14133)<br>
In ICLR 2023.<br>
> **Abstract:** *Deep generative models (DGMs) are data-eager because learning a complex model on limited data suffers from a large variance and easily overfits. Inspired by the classical perspective of the bias-variance tradeoff, we propose regularized deep generative model (Reg-DGM), which leverages a nontransferable pre-trained model to reduce the variance of generative modeling with limited data. Formally, Reg-DGM optimizes a weighted sum of a certain divergence and the expectation of an energy function, where the divergence is between the data and the model distributions, and the energy function is defined by the pre-trained model w.r.t. the model distribution. We analyze a simple yet representative Gaussian-fitting case to demonstrate how the weighting hyperparameter trades off the bias and the variance. Theoretically, we characterize the existence and the uniqueness of the global minimum of Reg-DGM in a non-parametric setting and prove its convergence with neural networks trained by gradient-based methods. Empirically, with various pre-trained feature extractors and a data-dependent energy function, Reg-DGM consistently improves the generation performance of strong DGMs with limited data and achieves competitive results to the state-of-the-art methods.*

## Requirements

* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory.
* CUDA toolkit 10.1 or later, and PyTorch 1.7.1 with compatible CUDA toolkit and torchvison.
* Python libraries: `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3 psutil scipy tensorboard ftfy regex scipy psutil matplotlib dill timm`.
* Install CLIP: `pip install git+https://github.com/openai/CLIP.git`.
* Install FaceNet: `pip install facenet-pytorch`.


## Dataset Preparation

**Our dataset preparation is the same as that of [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)**. Please refer to [stylegan2-ada-pytorch Preparing datasets](https://github.com/NVlabs/stylegan2-ada-pytorch#preparing-datasets) to download and prepare the datasets such as FFHQ.

## Training New Networks

Training the Reg-StyleGAN2
```python
# FFHQ-5k
python train.py --outdir=./output/ffhq-5k --data=/path/dataset/ffhq256x256.zip --cfg=paper256 --batch=64 \ 
--gpus=8 --subset=5000 --kimg=5000 --aug=noaug --metrics=fid50k_full \
--lamd=50 --pre_model=clip 

#CIFAR10
python train.py --outdir=./output/cifar10 --data=/path/dataset/cifar10.zip --cfg=cifar --batch=64 --gpus=8 \
--kimg=25000 --aug=noaug --metrics=fid50k_full --lamd=1e-5 --pre_model=resnet18 

```
Training the Reg-ADA
```python
# FFHQ-5k
python train.py --outdir=./output/ffhq-5k --data=/path/dataset/ffhq256x256.zip --cfg=paper256 --batch=64 \ 
--gpus=8 --subset=5000 --kimg=16000 --aug=ada --metrics=fid50k_full \
--lamd=1 --pre_model=clip 

#CIFAR-10
python train.py --outdir=./output/cifar10 --data=/path/dataset/cifar10.zip --cfg=cifar --batch=64 --gpus=8 \
--kimg=60000 --aug=ada --metrics=fid50k_full --lamd=5e-6 --pre_model=resnet18 

```

Some hyperparameters:

* `--cfg` (Default: `auto`) represents training configurations. Fllowing [ADA](https://github.com/NVlabs/stylegan2-ada-pytorch#training-new-networks), we use "--cfg=paper256" for the 256x256 resolution and "--cfg=cifar" for CIFAR10 dataset. 
* `--lamd` (Default: `1.0`) tradeoffs pre-trained models and generative models. The optimal lamd is relative to the used dataset and the pre-trained model. Too large lamd will cause the performance of the generative model to deteriorate.
* `--pre_model` (Default: `resnet18`) indicates used pre-trained model such as CLIP for 'clip' and FaceNet for 'facenet'.
* `--subset` controls the number of training images. If it is not specified, we will use the full training images.
* `--kimg` (Default: `25000`) controls the training length, representing how many real images are fed to the discriminator.

Please refer to [ADA](https://github.com/NVlabs/stylegan2-ada-pytorch#training-new-networks) for more hyperparameters.

## Evaluation Metrics

After the training, we can can compute metrics:

```python
python calc_metrics.py --metrics=fid50k_full,kid50k_full --data=/path/dataset/dataset.zip \ 
        --network=/path/checkpoint/network.pkl
```

The command above calculates the FID and KID metrics between the corresponding original full dataset and 50,000 generated images for a specified checkpoint pickle file. Please refer to [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch#quality-metrics) for more information.


## Inference for Generating Images

We can randomly generate images without fixed seeds by a pre-trained generative model stored as a `*.pkl` file:

```python
# Generate images with the truncation of 0.7
python random_generate.py --outdir=out --trunc=0.7 \ 
        --network=/path/checkpoint/network.pkl --images=100
```
Hyperparameter 'images' controlls the the number of generative images.

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{zhong2022deep,
  title={Deep Generative Modeling on Limited Data with Regularization by Nontransferable Pre-trained Models},
  author={Zhong, Yong and Liu, Hongtao and Liu, Xiaodong and Bao, Fan and Shen, Weiran and Li, Chongxuan},
  journal={arXiv preprint arXiv:2208.14133},
  year={2022}
}
```

## Acknowledgments

The code is developed based on [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch). We appreciate the nice PyTorch implementation.

## License

Copyright (c) 2022. All rights reserved.

The code is released under the [NVIDIA Source Code License](./LICENSE.txt).
