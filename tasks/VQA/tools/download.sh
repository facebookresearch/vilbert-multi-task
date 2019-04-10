## Script for downloading data

# # GloVe Vectors
# wget -P data http://nlp.stanford.edu/data/glove.6B.zip
# unzip data/glove.6B.zip -d data/glove
# rm data/glove.6B.zip

# Questions
# wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
# unzip data/v2_Questions_Train_mscoco.zip -d data
# rm data/v2_Questions_Train_mscoco.zip

# wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
# unzip data/v2_Questions_Val_mscoco.zip -d data
# rm data/v2_Questions_Val_mscoco.zip

# wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
# unzip data/v2_Questions_Test_mscoco.zip -d data
# rm data/v2_Questions_Test_mscoco.zip

# Annotations
# wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
# unzip data/v2_Annotations_Train_mscoco.zip -d data
# rm data/v2_Annotations_Train_mscoco.zip

# wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
# unzip data/v2_Annotations_Val_mscoco.zip -d data
# rm data/v2_Annotations_Val_mscoco.zip

# # Image Features
wget -P data https://imagecaption.blob.core.windows.net/imagecaption/trainval_36.zip
unzip data/trainval_36.zip -d data
rm data/trainval_36.zip
