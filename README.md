# multi-modal-bert
code and experiment for multi-modal bert to learn the vision and language representations. 


## Repository Setup

1. Create a fresh conda environment, and install all dependencies.

```text
conda create -n mmbert python=3.6
conda activate mmbert
pip install -r requirements.txt
```

2. Install this codebase as a package in this environment.

```text
python setup.py develop
```


# Data Setup

```
git clone https://github.com/jiasenlu/multi-modal-bert
cd multi-modal-bert
mkdir data
cd data
mkdir conceptual_caption
ln -s /srv/share2/jlu347/conceptual_caption/training_2 ./conceptual_caption/training
ln -s /srv/share2/jlu347/conceptual_caption/validation ./conceptual_caption/validation
```

To train the model: 

```
python train.py --do_train --num_workers 16 --from_pretrained --train_batch_size 512
```


# VQA Data Setup

```
ln -s /srv/share2/jlu347/conceptual_caption/VQA ./
```

# FOIL Data Setup

```
ln -s /srv/share/datasets/foil data/foil
ln -s /srv/share2/kd/multi-modal-bert/data/coco data/coco
```
