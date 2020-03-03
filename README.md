# Deep Fusion of Local and Non-Local Features for Precision Landslide Mapping

## Introduction
We proposes an effective approach to fuse both local and nonlocal features for precision landslide mapping, which achieves state-of-the-art segmentation performance for the landslide using images covering post-earthquake Jiuzhaigou.

## Testing set result

We test our DA-U-Net with more than 1000 UAV images convering Jiuzhaigou. We have released one sample image in the paper to reproduce the results.
Due to regulatory issues of China, we must remove the geoference information and downsample the image to half the size.

## Usage

  - The code is tested on miniconda3:4.7.10-cuda10.1-cudnn7-ubuntu18.04, please install tensorflow-gpu(version 1.9.0), Anaconda, CUDA9.0,CUDNN7.0 from source.
  - Install the dependencies using pip
   ```
   pip install -r requirements.txt
   ```
  - Or install the dependencies using anaconda
  ```bash
  conda install -y --channel=https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main --channel=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
        numpy==1.18.1 \
        opencv==3.4.2 \
        python==3.6.0 \
        tensorflow-gpu==1.9.0 \
    && conda clean -ya
  - If an error occurs, RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-dev=2.56.4-0ubuntu0.18.04.4 \
        python-qt4=4.12.1+dfsg-2 \
    && rm -rf /var/lib/apt/lists/*

  - run test code using `python -u 'test.py'`
  ```

## Dataset and model

The UAV images covering six town were obtained for the landslides caused by the earthquake in Jiuzhaigou, China on August 8, 2017.

# Reproduction

The code is also on [Code Ocean](https://codeocean.com/capsule/0157338/tree/v1) for the convinience of reproduction.

## Cite

```
@article{zhu2020deep,
  title={Deep Fusion of Local and Non-Local Features for Precision Landslide Recognition},
  author={Zhu, Qing and Chen, Lin and Hu, Han and Xu, Binzhi and Zhang, Yeting and Li, Haifeng},
  journal={arXiv preprint arXiv:2002.08547},
  year={2020}
}
```
