#!/bin/bash

#download weights for nets

OUTPUT_PATH=../weights
mkdir -p ${OUTPUT_PATH}

#for detection
DETECTION_NET='1bplXUJ4t-5GtI5K3ZCPAiNFaEQ5QjbXk'
gdown --id ${DETECTION_NET} --fuzzy --output ${OUTPUT_PATH}/htc.pth

#for pointtrack tracker we have 2 options of weights
#for pigs1 embeddings
PIGS1='1cjmJqhzfhFdUWESKvmigeio-YGTm-KpL'
gdown --id ${PIGS1} --fuzzy --output ${OUTPUT_PATH}/pigs1.pth

#for pigs2 embeddings
PIGS2='1FWBIjOdaxtrNDFGhICYOWw5HIG_oVrEc'
gdown --id ${PIGS2} --fuzzy --output ${OUTPUT_PATH}/pigs2.pth
