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

   Download the data from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), and place the downloaded data in the folder`./dataset`. You can also download the STL results from [[Google Drive]](https://drive.google.com/file/d/1gWliIGDDSi2itUAvYaRgACru18j753Kw/view?usp=sharing), and place the downloaded data in the folder`./stl`.

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

# TETS dataset

Here is the prompts use to generate the coresponding textual informaton of time series via [[OPENAI ChatGPT-3.5 API]](https://platform.openai.com/docs/guides/text-generation)

<div align="center"><img src=./pics/TETS_prompt.png, width=80% /></div>

The time series data are come from [[S&P 500]](https://www.spglobal.com/spdji/en/indices/equity/sp-500/#overview).

You can download the processed data with text embedding from GPT2 from: [[TETS]](https://drive.google.com/file/d/1Hu2KFj0kp4kIIpjbss2ciLCV_KiBreoJ/view?usp=drive_link
)



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