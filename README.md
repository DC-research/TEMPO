# Time Series Foundation Model - TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting


<div align="center"><img src=./pics/TEMPO_logo.png width=80% /></div>

The official code for [["TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting (ICLR 2024)"]](https://arxiv.org/pdf/2310.04948).

TEMPO is one of the very first open source **Time Series Foundation Models** for forecasting task v1.0 version.

<div align="center"><img src=./pics/TEMPO.png width=80% /></div>

## Demos

### 1. Reproducing zero-shot experiments on ETTh2:

Please try to reproduc the zero-shot experiments on ETTh2 [[here on Colab]](https://colab.research.google.com/drive/11qGpT7H1JMaTlMlm9WtHFZ3_cJz7p-og?usp=sharing).

### 2. Zero-shot experiments on customer dataset:

We use the following Colab page to show the demo of building the customer dataset and directly do the inference via our pre-trained foundation model: [[Colab]](https://colab.research.google.com/drive/1ZpWbK0L6mq1pav2yDqOuORo4rHbv80-A?usp=sharing)

### 3. Online demo:

Please try our foundation model demo [[here]](https://4171a8a7484b3e9148.gradio.live).

<div align="center"><img src=./pics/TEMPO_demo.jpg width=80% /></div>

## Practice on your end

We also updated our models on HuggingFace: [[Melady/TEMPO]](https://huggingface.co/Melady/TEMPO).

### Build the environment

```
conda create -n tempo python=3.8
```
```
conda activate tempo
```
```
pip install -r requirements.txt
```

### Get Data

   Download the data from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing) or [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy), and place the downloaded data in the folder`./dataset`. You can also download the STL results from [[Google Drive]](https://drive.google.com/file/d/1gWliIGDDSi2itUAvYaRgACru18j753Kw/view?usp=sharing), and place the downloaded data in the folder`./stl`.

### Run TEMPO

### Pre-Training Stage
```
bash [ecl, etth1, etth2, ettm1, ettm2, traffic, weather].sh
```

### Test/ Inference Stage

After training, we can test TEMPO model under the zero-shot setting:

```
bash [ecl, etth1, etth2, ettm1, ettm2, traffic, weather]_test.sh
```

<div align="center"><img src=./pics/results.jpg width=90% /></div>


## Pre-trained Models

You can download the pre-trained model from [[Google Drive]](https://drive.google.com/file/d/11Ho_seP9NGh-lQCyBkvQhAQFy_3XVwKp/view?usp=drive_link) and then run the test script for fun.

## TETS dataset

Here is the prompts use to generate the coresponding textual informaton of time series via [[OPENAI ChatGPT-3.5 API]](https://platform.openai.com/docs/guides/text-generation)

<div align="center"><img src=./pics/TETS_prompt.png width=80% /></div>

The time series data are come from [[S&P 500]](https://www.spglobal.com/spdji/en/indices/equity/sp-500/#overview). Here is the EBITDA case for one company from the dataset:


<div align="center"><img src=./pics/Company1_ebitda_summary.png width=80% /></div>

Example of generated contextual information for the Company marked above:

<div align="center"><img src=./pics/Company1_ebitda_summary_words.jpg width=80% /></div>




You can download the processed data with text embedding from GPT2 from: [[TETS]](https://drive.google.com/file/d/1Hu2KFj0kp4kIIpjbss2ciLCV_KiBreoJ/view?usp=drive_link
).

## Contact
Feel free to connect DefuCao@USC.EDU / YanLiu.CS@USC.EDU if you’re interested in applying TEMPO to your real-world application.

## Cite our work
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

```
@article{
   Jia_Wang_Zheng_Cao_Liu_2024, 
   title={GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting}, 
   volume={38}, 
   url={https://ojs.aaai.org/index.php/AAAI/article/view/30383}, 
   DOI={10.1609/aaai.v38i21.30383}, 
   number={21}, 
   journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
   author={Jia, Furong and Wang, Kevin and Zheng, Yixiang and Cao, Defu and Liu, Yan}, 
   year={2024}, month={Mar.}, pages={23343-23351} 
   }
```
