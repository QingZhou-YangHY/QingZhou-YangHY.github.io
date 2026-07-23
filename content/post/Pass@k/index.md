---
title: "Pass@k论文复现"
description: "Pass@k"
date: 2025-09-21T16:00:00+08:00
slug: 
image:  'ByteDance.jpg'
math: false
license: 
hidden: false
comments: true
draft: false
categories:
    - Notes

--- 
![](Seed.jpg "Seed")
# Pass@k Training for Adptively Balancing Eplortion and Exploitation of LRMs

## Pass@k论文解读  

policy探索能力的指标：the natural prevention of the decrease in the entropy of policy distribution  

Pass@1和Pass@k之间的主要区别在于奖励计算和优势估计过程。

vanilla：原来的  
Ground Truth是正确答案（标准答案）

#### 探索能力:

- the entropy of policy distribution处在一个较高的水平  

- the answer diversity of the negative response处在一个较高的水平  
Pass@k的entropy在RLVR procedure的200step左右开始上升  

### k值影响Pass@k训练  

实验中从4，8，16中调整k值

无论k的值是多少，随着训练的进行，训练奖励都可以提高到相对较高的水平，这表明k的值不是帮助LLMs逃脱局部最优解的关键性因素  

#### k值越大，训练效率越慢（k值越大，优势值越小，导致优化步骤越短，训练效率越低）  

### 训练效率的影响因素  

实验在N = 32 和 k = 8的设置下使用{1 x 10-6,2 x 10-6,4 x 10-6}learning rate  

#### 随着学习率的提高，拐点出现得更早，表明训练效率更高

#### 最大优势值不是帮助模型表现优异的关键性因素

#### 为更难的问题分配更大的优化强度可以有效地提高训练效率  

根据the entropy of policy distribution的高低，区分high-exploration和low-exploration。  

high-exploration使用Pass@1 advantage fuction来exploit prior exploration，low-exploration使用Pass@k advantage function来encourage further exploration。  

#### Implicit reward design（隐式奖励设计）可以控制优化过程

具体地说，结合或动态调整不同形式的优势估计，可以同时提高exploration and exploitation能力  

---

## Pass@k论文复现

[arxiv文章](https://arxiv.org/abs/2508.10751)👈  
[github仓库](https://github.com/RUCAIBox/Passk_Training)👈  
[Datasets](https://huggingface.co/datasets/RUC-AIBOX/Passk_Training_Maze)👈  

### Maze
每个迷宫由文本表示，包含n行和n列，总共n∗n个字符  

四个字符“S”、“E”、“.”和“*”中的一个，分别表示起点、目的地、可用地点和不可用地点  

给定迷宫，LLM可以首先生成思维或推理过程，然后生成最终答案，其中包括四个动作“U”、“D”、“L”和“R”中的一个，分别表示向上、向下、向左和向右移动  

对于训练数据，我们构建了大小为9×9、11×11、13×13和15×15的迷宫，以增加训练数据的多样性  

对于测试数据，为了评估RLVR过程的泛化能力，我们不仅使用训练数据集进行相同大小的迷宫，还收集了大小为7×7、17×17、19×19和21×21的迷宫  

为了确保实验的有效性，我们在生成训练和测试数据后进行了严格的重复数据删除操作  

Training Set 都是10,000;Test Set除了7 * 7,剩下的都是100  

### Implementation Details
#### Training
backbone model:Qwen2.5-7B-Instruct和Qwen2.5-32B Instruct  

DAPO  

εlow=0.2和εhigh=0.28  

token-level policy gradient loss  

remove other optimizations  

learning rate:1 × 10−6  

warmup:10  

prompt batch size(BS prompt):128  

- Prompt Batch Size：是模型推理生成的粒度。它决定在一次性并行处理多少个独立的提示词（例如，多少个用户问题），并同时为它们生成文本  

mini-batch size(BS mini):32  

- Mini-batch Size：是模型权重更新的粒度。它决定在计算一次梯度下降时，使用多少条训练数据  

rollout times:32  

positive reward Rpos = 1  

negative reward Rneg = 0  

do not employ any regularization methods, such as KL or Entropy regularization  
temperature:1.0  

Top_P:0.95  

- 从概率最高的词开始累加它们的概率，直到累积概率达到或刚刚超过你设定的 top_p 值

For each question,we sample 32responses for Maze task and sample 8 responses for other tasks  

### Versions/Dependencies
Python 3.10.18  
Ray 2.49.1  
grpcio 1.75.0  
Ubuntu 24.04.2 LTS  

### 如何从huggingface上下载数据集和模型  
从huggingface上下载文件有2种方式，一种是直接登录后在网页上下载；一种是通过huggingface-cli命令下载。  

本文介绍的是第二种下载方式。
### 安装
对于huggingface-cli命令的下载直接通过pip命令安装即可：  
> pip install -U huggingface_hub[hub_transfer]  

对于国内用户还可以通过设置镜像网站的方式加速下载：  
> #linux
export HF_ENDPOINT=https://hf-mirror.com  
#windows  
set HF_ENDPOINT=https://hf-mirror.com  

使用命令行下载  
#### 模型
> huggingface-cli download --resume-download [1] --local-dir [2] --local-dir-use-symlinks False
#### 数据集
> huggingface-cli download --repo-type dataset --resume-download [3] --local-dir [4] --local-dir-use-symlinks False --token hf_***

格式为：[1]和[3]表示项目的路径，格式为用户名/项目，比如mistralai/Mistral-7B-Instruct-v0.2表示的是mistralai下的7B instruct v0.2权重。[2]和[4]表示的是本地的保存地址。

需要的注意的是有些仓库需要登录才可以下载，形如–token hf_***为huggingface的token配置。token的生成需要在huggingface个人页面生成.