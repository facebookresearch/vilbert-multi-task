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

Download the data for different datasets to the `data` directory.

