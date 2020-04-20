# Data Setup


## Extracting features

1. Install [`vqa-maskrcnn-benchmark`](https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark) repository and download the model and config. 

```text
cd data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
```


2. Extract features for images

Run from root directory

```text
python script/extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --image_dir <path_to_directory_with_images> --output_folder <path_to_output_extracted_features>
```

3. Extract features for images with GT bbox

Generate a `.npy` file with the following format for all the images and their bboxes

```text
{
    {
        'file_name': 'name_of_image_file',
        'file_path': '<path_to_image_file_on_your_disk>',
        'bbox': array([
                        [ x1, y1, width1, height1],
                        [ x2, y2, width2, height2],
                        ...
                    ]),
        'num_box': 2
    },
    ....
}
```

Run from root directory

```text
python script/extract_features.py --model_file data/detectron_model.pth --config_file data/detectron_config.yaml --imdb_gt_file <path_to_imdb_npy_file_generated_above> --output_folder <path_to_output_extracted_features>
```

4. Convert the extracted images to an LMDB file

```text
python script/convert_to_lmdb.py --features_dir <path_to_extracted_features> --lmdb_file <path_to_output_lmdb_file>
```

## Datasets

Download the data for different datasets to the `data` directory. Here are the links for downloading all the data for *downstream* tasks used in this project :

1. Run from root directory

```text
cd data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/datasets.tar.gz
tar xf datasets.tar.gz
```

The extracted folder has all the datasets and their cache directories that can be pointed to in the `vilbert_tasks.yaml` file.

2. Download extracted features for COCO, GQA and NLVR2

Some of the features are not present in the extracted folder in Step 1. Those can be downloaded following these commands :

#### COCO features

```text
cd coco

mkdir features_100

cd features_100

mkdir COCO_test_resnext152_faster_rcnn_genome.lmdb

mkdir COCO_trainval_resnext152_faster_rcnn_genome.lmdb

wget https://dl.fbaipublicfiles.com/vilbert-multi-task/datasets/coco/features_100/COCO_trainval_resnext152_faster_rcnn_genome.lmdb/data.mdb && mv data.mdb features_100/COCO_trainval_resnext152_faster_rcnn_genome.lmdb/

wget https://dl.fbaipublicfiles.com/vilbert-multi-task/datasets/coco/features_100/COCO_test_resnext152_faster_rcnn_genome.lmdb/data.mdb && mv data.mdb features_100/COCO_test_resnext152_faster_rcnn_genome.lmdb/
```

#### GQA features

```text
cd gqa

mkdir gqa_resnext152_faster_rcnn_genome.lmdb

cd gqa_resnext152_faster_rcnn_genome.lmdb

wget https://dl.fbaipublicfiles.com/vilbert-multi-task/datasets/gqa/gqa_resnext152_faster_rcnn_genome.lmdb/data.mdb
```

#### NLVR2 features

```text
cd nlvr2

mkdir nlvr2_resnext152_faster_rcnn_genome.lmdb

cd nlvr2_resnext152_faster_rcnn_genome.lmdb

wget https://dl.fbaipublicfiles.com/vilbert-multi-task/datasets/nlvr2/nlvr2_resnext152_faster_rcnn_genome.lmdb/data.mdb
```
