# Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective

## Overview
This repository contains the code to run the experiments present in this paper: [Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective](https://arxiv.org/abs/2402.18607). 

**Remarks:** 
1. This repository includes a functional framework for the property inference attack (PIA) and fairness poisoning attack (FPA) in the paper. However, the code for SDEM is not included in this repository as it runs on a separate environment and is not compatible with this repository.
1. Some folders may need to be manually created during running. Please follow the error information to create necessary folders.
1. The authors did not carefully check the correctness of this repository at a new environment, so bugs should be expected. If you come across any errors or bugs that cannot be resolved, please contact [the first author](https://xinjianluo.github.io/).


## How to run
### Step 1: Install dependencies
We suggest creating a new virtual environment in Anaconda and install necessary libs in this environment.

The libs include but are not constrained to [CLIP](https://github.com/openai/CLIP), [Mimicry](https://github.com/kwotsin/mimicry/tree/master), and [NPEET](https://github.com/gregversteeg/NPEET).


### Step 2: Run FPA in 
    main-FPA.py

### Step 2: Run PIA in
    main-PIA.ipynb

    

## Citation
If you use our results or this codebase in your research, then please cite this paper:
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