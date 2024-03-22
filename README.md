# TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting

The official code for "TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting (ICLR 2024)".


<div align="center"><img src=./pics/TEMPO.png width=80% /></div>


# Build the environment

```
conda create -n tempo python=3.8
```
```
conda activate tempo
```
```
pip install -r requirements.txt
```

# Get Data

   Download the data from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), and place the downloaded data in the folder`./dataset`

# Run TEMPO

## Training Stage
```
bash [ecl, etth1, etth2, ettm1, ettm2, traffic, weather].sh
```

## Test

```
bash [ecl, etth1, etth2, ettm1, ettm2, traffic, weather]_test.sh
```

# Pre-trained Models

You can download the pre-trained model from [[Google Drive]](https://drive.google.com/file/d/11Ho_seP9NGh-lQCyBkvQhAQFy_3XVwKp/view?usp=drive_link) and then run the test script for fun.

## Cite
```
@inproceedings{
cao2024tempo,
title={{TEMPO}: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting},
author={Defu Cao and Furong Jia and Sercan O Arik and Tomas Pfister and Yixiang Zheng and Wen Ye and Yan Liu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=YH5w12OUuU}
}
```