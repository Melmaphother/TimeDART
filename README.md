<div align="center">
  <!-- <h1><b> TimeDART </b></h1> -->
  <!-- <h2><b> TimeDART </b></h2> -->
  <h2>
    <b> 
      <img src="assets/dart.png" alt="Dart Image" style="vertical-align: baseline; width: 2em; height: 2em; margin-bottom: -0.2em;">
      TimeDART: A Diffusion Autoregressive Transformer <br> for Self-Supervised Time Series Representation 
    </b>
  </h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/Melmaphother/TimeDART?color=green)
![](https://img.shields.io/github/stars/Melmaphother/TimeDART?color=yellow)
![](https://img.shields.io/github/forks/Melmaphother/TimeDART?color=lightblue)

</div>

---

>🙋 Please let us know if you find out a mistake or have any suggestions!
>
>🌟 If you find our work helpful, please consider to star this repository and cite our research:

```bibtex
@misc{wang2025timedart,
      title={TimeDART: A Diffusion Autoregressive Transformer for Self-Supervised Time Series Representation}, 
      author={Daoyu Wang and Mingyue Cheng and Zhiding Liu and Qi Liu and Enhong Chen},
      year={2025},
      eprint={2410.05711},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.05711}, 
}
```



## Updates/News:

🚩 **News** (Jan. 2025): TimeDART is under review.

🚩 **News** (Oct. 2024): TimeDART initialized. "Dart" means a small, pointed object 🎯 thrown to hit a target accurately.

## Introduction

Self-supervised learning has garnered increasing attention in time series analysis for benefiting various downstream tasks and reducing reliance on labeled data. Despite its effectiveness, existing methods often struggle to comprehensively capture both long-term dynamic evolution and subtle local patterns in a unified manner. In this work, we propose \textbf{TimeDART}, a novel self-supervised time series pre-training framework that unifies two powerful generative paradigms to learn more transferable representations. Specifically, we first employ a \textit{causal} Transformer encoder, accompanied by a patch-based embedding strategy, to model the evolving trends from left to right. Building on this global modeling, we further introduce a denoising diffusion process to capture fine-grained local patterns through forward diffusion and reverse denoising. Finally, we optimize the model in an autoregressive manner. As a result, TimeDART effectively accounts for both global and local sequence features in a coherent way. We conduct extensive experiments on public datasets for time series forecasting and classification. The experimental results demonstrate that TimeDART consistently outperforms previous compared methods, validating the effectiveness of our approach.

![](assets/model.png)

## Performance

### Forecasting

![](assets/table1.png)

### Classification

![](assets/table2.png)

## Usage

Datasets used in our experiments can be found in [Google Drive](https://drive.google.com/drive/folders/19P---oV4nQ53JgKnE0VX3t_N1jLliVSv?usp=drive_link).

```sh
cd TimeDART
conda create -n timedart python=3.10
pip install -r requirements.txt
```

We provide the default hyper-parameter settings in `scripts/prertrain` to perform pretraining, and ready-to-use scripts for fine-tuning on each datasets in `scripts/finetune`.

```sh
sh scripts/pretrain/ETTh2.sh && sh scripts/finetune/ETTh2.sh
```

## Acknowledgement

This repo is built on the pioneer works. We appreciate the following GitHub repos a lot for their valuable code base or datasets:

[Time-Series-Library](https://github.com/thuml/Time-Series-Library?tab=readme-ov-file)

[GPHT](https://github.com/icantnamemyself/GPHT/tree/main)
