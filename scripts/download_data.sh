#!/bin/bash

#download the dataset
OUTPUT_PATH=../data/pigs/
mkdir -p ${OUTPUT_PATH}

# combined dataset
DATASET='1aGdnLGcv28N9wszNRWil4QUp_sckVVMZ'
gdown --id ${DATASET} --fuzzy --output ${OUTPUT_PATH}/data.zip
unzip ${OUTPUT_PATH}/data.zip -d ${OUTPUT_PATH}
rm ${OUTPUT_PATH}/data.zip
