# Time Series Foundation Model - TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=2310.04948&color=B31B1B&logo=arXiv)](https://arxiv.org/pdf/2310.04948)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/Melady/TEMPO)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
</div>

<div align="center"><img src=https://raw.githubusercontent.com/DC-research/TEMPO/main/tempo/pics/TEMPO_logo.png width=80% /></div>

The official code for [["TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting (ICLR 2024)"]](https://arxiv.org/pdf/2310.04948).

TEMPO is one of the very first open source **Time Series Foundation Models** for forecasting task v1.0 version.

<div align="center"><img src=https://raw.githubusercontent.com/DC-research/TEMPO/main/tempo/pics/TEMPO.png width=80% /></div>

## ⏳ Upcoming Features

- [✅] Parallel pre-training pipeline
- [] Probabilistic forecasting
- [] Multimodal dataset
- [] Multimodal pre-training script
	

## 🚀 News

- **Nov 2024**: 🚀 We've published **TimeAGI** on PyPI! Now you can simply `pip install timeagi` to get started and try **TEMPO** by `from tempo.models.TEMPO import TEMPO`. Check out our demo for more details: [TimeAGI](https://pypi.org/project/timeagi/)!

- **Oct 2024**: 🚀 We've streamlined our code structure, enabling users to download the pre-trained model and perform zero-shot inference with a single line of code! Check out our [demo](./run_TEMPO_demo.py) for more details. Our model's download count on HuggingFace is now trackable!

- **Jun 2024**: 🚀 We added demos for reproducing zero-shot experiments in [Colab](https://colab.research.google.com/drive/11qGpT7H1JMaTlMlm9WtHFZ3_cJz7p-og?usp=sharing).  We also added the demo of building the customer dataset and directly do the inference via our pre-trained foundation model: [Colab](https://colab.research.google.com/drive/1ZpWbK0L6mq1pav2yDqOuORo4rHbv80-A?usp=sharing)
- **May 2024**: 🚀 TEMPO has launched a GUI-based online [demo](https://4171a8a7484b3e9148.gradio.live/), allowing users to directly interact with our foundation model!
- **May 2024**: 🚀 TEMPO published the 80M pretrained foundation model in [HuggingFace](https://huggingface.co/Melady/TEMPO)!
- **May 2024**: 🧪 We added the code for pretraining and inference TEMPO models.  You can find a pre-training script demo in [this folder](./scripts/etth2.sh). We also added [a script](./scripts/etth2_test.sh) for the inference demo.

- **Mar 2024**: 📈  Released [TETS dataset](https://drive.google.com/file/d/1Hu2KFj0kp4kIIpjbss2ciLCV_KiBreoJ/view?usp=drive_link) from [S&P 500](https://www.spglobal.com/spdji/en/indices/equity/sp-500/#overview) used in multimodal experiments in TEMPO. 
- **Mar 2024**: 🧪 TEMPO published the project [code](https://github.com/DC-research/TEMPO) and the pre-trained checkpoint [online](https://drive.google.com/file/d/11Ho_seP9NGh-lQCyBkvQhAQFy_3XVwKp/view?usp=drive_link)! 
- **Jan 2024**: 🚀 TEMPO [paper](https://openreview.net/pdf?id=YH5w12OUuU) get accepted by ICLR!
- **Oct 2023**: 🚀 TEMPO [paper](https://arxiv.org/pdf/2310.04948) released on Arxiv!

## Build the environment

```
conda create -n tempo python=3.8
```
```
conda activate tempo
```
```
pip install timeagi
```

## Script Demo

A streamlining example showing how to perform forecasting using TEMPO:

```python
# Third-party library imports
import numpy as np
import torch
from numpy.random import choice
# Local imports
from tempo.models.TEMPO import TEMPO


model = TEMPO.load_pretrained_model(
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        repo_id = "Melady/TEMPO",
        filename = "TEMPO-80M_v1.pth",
        cache_dir = "./checkpoints/TEMPO_checkpoints"  
)

input_data = np.random.rand(336)    # Random input data
with torch.no_grad():
        predicted_values = model.predict(input_data, pred_length=96)
print("Predicted values:")
print(predicted_values)

```

## Demos

### 1. Reproducing zero-shot experiments on ETTh2:

Please try to reproduc the zero-shot experiments on ETTh2 [[here on Colab]](https://colab.research.google.com/drive/11qGpT7H1JMaTlMlm9WtHFZ3_cJz7p-og?usp=sharing).

### 2. Zero-shot experiments on customer dataset:

We use the following Colab page to show the demo of building the customer dataset and directly do the inference via our pre-trained foundation model: [[Colab]](https://colab.research.google.com/drive/1ZpWbK0L6mq1pav2yDqOuORo4rHbv80-A?usp=sharing)

### 3. Online demo:

Please try our foundation model demo [[here]](https://4171a8a7484b3e9148.gradio.live).

<div align="center"><img src=https://raw.githubusercontent.com/DC-research/TEMPO/main/tempo/pics/TEMPO_demo.jpg width=80% /></div>

## Practice on your end

We also updated our models on HuggingFace: [[Melady/TEMPO]](https://huggingface.co/Melady/TEMPO).



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

<div align="center"><img src=https://raw.githubusercontent.com/DC-research/TEMPO/main/tempo/pics/results.jpg width=90% /></div>


## Pre-trained Models

You can download the pre-trained model from [[Google Drive]](https://drive.google.com/file/d/11Ho_seP9NGh-lQCyBkvQhAQFy_3XVwKp/view?usp=drive_link) and then run the test script for fun.

## TETS dataset

Here is the prompts use to generate the coresponding textual informaton of time series via [[OPENAI ChatGPT-3.5 API]](https://platform.openai.com/docs/guides/text-generation)

<div align="center"><img src=https://raw.githubusercontent.com/DC-research/TEMPO/main/tempo/pics/TETS_prompt.png width=80% /></div>

The time series data are come from [[S&P 500]](https://www.spglobal.com/spdji/en/indices/equity/sp-500/#overview). Here is the EBITDA case for one company from the dataset:


<div align="center"><img src=https://raw.githubusercontent.com/DC-research/TEMPO/main/tempo/pics/Company1_ebitda_summary.png width=80% /></div>

Example of generated contextual information for the Company marked above:

<div align="center"><img src=https://raw.githubusercontent.com/DC-research/TEMPO/main/tempo/pics//Company1_ebitda_summary_words.jpg width=80% /></div>




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
