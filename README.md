# Multi-Task Vision and Language Representation Learning (ViLBERT-MT)

Code and pre-trained models for **12-in-1: Multi-Task Vision and Language Representation Learning**.


## Repository Setup

1. Create a fresh conda environment, and install all dependencies.

```text
conda create -n vilbert python=3.6
conda activate vilbert-MT
git clone xxx
cd ViLBert-MT
pip install -r requirements.txt
```

2. Install pytorch
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

3. Install apex, follows https://github.com/NVIDIA/apex

4. Install this codebase as a package in this environment.
```text
python setup.py develop
```

## Data Setup

Check `README.md` under `data` for more details.  

## Visiolinguistic Pre-training

To train the model: 

```
To be added
```

For internal use: copy the pre-trained checkpoint from Skynet 

```
cp -a /srv/share3/jlu347/vilbert-MT/save/* #to_your_directory.
```

## Benchmark Vision-Lanugage Tasks 

| Task    | Sub-Task | Model | LR   | Results (split) |
|:-------:|:------:|:---:|:------:|:--------------------------------------:|
| **VQA** | - | **ViLBERT** | 4e-5 | **70.55** (test-dev) |
| - | - | DFAF | - |70.22 (test-dev) |
|**Ref Expression** | RefCOCO+ | **ViLBERT** | 4e-5 | **72.34** (val) - **78.52** (testA) - **62.61** (testB) |
|-|RefCOCO+|MAttNet|-|65.33 (val) - 71.62 (testA) - 56.02 (testB)|
|**Ref Expression**|RefCOCO|**ViLBERT**|4e-5|-|
|-|RefCOCO|MAttNet|-|-|
|**Ref Expression**|Refg|**ViLBERT**|4e-5|-|
|-|Refg|MAttNet|-|-|
|**Image Caption Ranking**|Image Retrieval|**ViLBERT**|2e-5|**58.20** (R1) - **84.90** (R5) - **91.52** (R10)|
|-|Image Retrieval|SCAN|-|48.60 (R1) - 77.70 (R5) - 85.20 (R10)|


## Single-task Training
### VQA 

To fintune a 6-layer vilbert model for VQA with 8 GPU. `--tasks 1` means VQA tasks. Check `vlbert_tasks.yml` for more settings for VQA tasks.  

```bash
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 train_tasks.py --bert_model bert-base-uncased --from_pretrained save/bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin  --config_file config/bert_base_6layer_6conect.json  --learning_rate 4e-5 --num_workers 16 --tasks 1 --save_name pretrained
```

### Refer Expression
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 train_tasks.py --bert_model bert-base-uncased --from_pretrained save/bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin  --config_file config/bert_base_6layer_6conect.json  --learning_rate 4e-5 --num_workers 16 --tasks 11 --save_name pretrained
```

### Image Retrieval
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 train_tasks.py --bert_model bert-base-uncased --from_pretrained save/bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin  --config_file config/bert_base_6layer_6conect.json  --learning_rate 4e-5 --num_workers 9 --tasks 11 --save_name pretrained
```

## Multi-task Training

```
code tobe added here.
```



## Fine-tune from Multi-task trained model

```
code tobe added here.
```



### Add your own tasks 


