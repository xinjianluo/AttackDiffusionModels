# Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective

## Overview
This repository contains the code to replicate the experiments presented in the paper: [Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective](https://arxiv.org/abs/2402.18607). 

**Remarks:** 
1. This repository includes a functional framework for the Property Inference Attack (PIA) and Fairness Poisoning Attack (FPA) described in the paper. However, the code for SDEM is not included, as it operates in a separate environment and is incompatible with this repository.
1. Some folders may need to be created manually during execution. Please follow the error messages to create the required folders.
1. The authors have not thoroughly tested the repository in a new environment, so bugs may be encountered. If you encounter errors or issues that cannot be resolved, please contact [the first author](https://xinjianluo.github.io/) for assistance.


## How to run
### Step 1: Install dependencies
We suggest creating a new virtual environment using Anaconda and installing necessary libs in this environment.

Required libraries include, but are not limited to, [CLIP](https://github.com/openai/CLIP), [Mimicry](https://github.com/kwotsin/mimicry/tree/master), and [NPEET](https://github.com/gregversteeg/NPEET).


### Step 2: Run FPA, execute 
    main-FPA.py

### Step 2: Run PIA, open and run
    main-PIA.ipynb

    

## Citation
If you use the results or codebase from this repository in your research, please cite the paper:
```
@article{LuoJWWXO24,
  author       = {Xinjian Luo and
                  Yangfan Jiang and
                  Fei Wei and
                  Yuncheng Wu and
                  Xiaokui Xiao and
                  Beng Chin Ooi},
  title        = {Exploring Privacy and Fairness Risks in Sharing Diffusion Models:
                  An Adversarial Perspective},
  journal      = {{IEEE} Trans. Inf. Forensics Secur.},
  volume       = {19},
  pages        = {8109--8124},
  year         = {2024}
}
```