# 12-in-1: Multi-Task Vision and Language Representation Learning

Please cite the following if you use this code. Code and pre-trained models for [12-in-1: Multi-Task Vision and Language Representation Learning](http://openaccess.thecvf.com/content_CVPR_2020/html/Lu_12-in-1_Multi-Task_Vision_and_Language_Representation_Learning_CVPR_2020_paper.html):

```
@InProceedings{Lu_2020_CVPR,
author = {Lu, Jiasen and Goswami, Vedanuj and Rohrbach, Marcus and Parikh, Devi and Lee, Stefan},
title = {12-in-1: Multi-Task Vision and Language Representation Learning},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

and [ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265):

```
@inproceedings{lu2019vilbert,
  title={Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks},
  author={Lu, Jiasen and Batra, Dhruv and Parikh, Devi and Lee, Stefan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={13--23},
  year={2019}
}
```

## Repository Setup

1. Create a fresh conda environment, and install all dependencies.

```text
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
git clone --recursive https://github.com/facebookresearch/vilbert-multi-task.git
cd vilbert-multi-task
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

## Visiolinguistic Pre-training and Multi Task Training

### Pretraining on Conceptual Captions

```
python train_concap.py --bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json --train_batch_size 512 --objective 1 --file_path <path_to_extracted_cc_features>
```
[Download link](https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin)

### Multi-task Training

```
python train_tasks.py --bert_model bert-base-uncased --from_pretrained <pretrained_model_path> --config_file config/bert_base_6layer_6conect.json --tasks 1-2-4-7-8-9-10-11-12-13-15-17 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name multi_task_model
```

[Download link](https://dl.fbaipublicfiles.com/vilbert-multi-task/multi_task_model.bin)


### Fine-tune from Multi-task trained model

```
python train_tasks.py --bert_model bert-base-uncased --from_pretrained <multi_task_model_path> --config_file config/bert_base_6layer_6conect.json --tasks 1 --lr_scheduler 'warmup_linear' --train_iter_gap 4 --task_specific_tokens --save_name finetune_from_multi_task_model
```
 
## License

vilbert-multi-task is licensed under MIT license available in [LICENSE](LICENSE) file.
