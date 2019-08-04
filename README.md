# ViLBERT
Code and experiment for multi-modal bert to learn the vision and language representations. 

## Repository Setup

1. Create a fresh conda environment, and install all dependencies.

```text
conda create -n vilbert python=3.6
conda activate vilbert
git clone https://github.com/jiasenlu/ViLBert
cd ViLBert
pip install -r requirements.txt
```

2. Install pytorch
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

3. Install apx, follows https://github.com/NVIDIA/apex

4. Install this codebase as a package in this environment.
```text
python setup.py develop
```

## Visiolinguistic Pre-training

```
mkdir data
cd data
mkdir conceptual_caption
ln -s /srv/share2/jlu347/conceptual_caption/training_2 ./conceptual_caption/training
ln -s /srv/share2/jlu347/conceptual_caption/validation ./conceptual_caption/validation
```

To train the model: 

```

```

## Vision-Lanugage Benchmark

Task    | Sub-Task | Model | LR   | Results (split) 
:-------:|:------:|:---:|:------:|:--------------------------------------:
 **VQA** | - | **ViLBERT** | 4e-5 | **70.55** (test-dev) 
 | - | DFAF | - |70.22 (test-dev) ||
**VCR**   | Q->A | **ViLBERT** | 2e-5 | **73.3** (test) 
|Q->A|R2C|-|63.8 (test)
**VCR** | QA->R | **ViLBERT** | 2e-5 | **74.6** (test) 
 | QA->R | R2C | - | 67.3 (test) 
**VCR** | Q->AR | **ViLBERT** | 2e-5 |   **54.8** (test)    
 | Q->AR | R2C | - | 44.0 (test) 
**Ref Expression** | RefCOCO+ | **ViLBERT** | 4e-5 | **72.34** (val) - **78.52** (testA) - **62.61** (testB) 
|RefCOCO+|MAttNet|-|65.33 (val) - 71.62 (testA) - 56.02 (testB)
**Ref Expression**|RefCOCO|**ViLBERT**|4e-5|-
|RefCOCO|MAttNet|-|-
**Ref Expression**|Refg|**ViLBERT**|4e-5|-
|Refg|MAttNet|-|-
**Image Caption Ranking**|Image Retrieval|**ViLBERT**|2e-5|**58.20** (R1) - **84.90** (R5) - **91.52** (R10)
|Image Retrieval|SCAN|-|48.60 (R1) - 77.70 (R5) - 85.20 (R10)
**Image Caption Ranking**|Caption Retrieval|**ViLBERT**|2e-5|-
|Caption Retrieval|SCAN|-|-


## TASK: VQA 

```

```

## TASK: VCR
```

```

## TASK: Refer Expression
```

```

## TASK: Image Retrieval
```

```

## Add your own tasks
```

```

