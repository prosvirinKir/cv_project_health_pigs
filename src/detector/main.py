#!/usr/bin/python3

import os
import yaml
import numpy as np

from mmdet.apis import (
    inference_detector,
    init_detector,
)

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''

class Detector:

    def __init__(
        self,
        config: str,
    ) -> None:
        
        assert os.path.exists(config), f'Config file {config} does not exist!'

        with open(config, 'r') as file:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)

        self.model = init_detector(
            kwargs['config'],
            kwargs['checkpoint'],
            kwargs['device'],
        )
        self.treshold = float(kwargs['treshold'])
        self.mask = np.load(kwargs['mask'])

    def detect(
        self,
        image: np.array,
    ) -> dict:
    
        ''' Inference Detector
        
        Inputs:
        image: np.array. Shape (HxWx3), BGR channel order, Not preprocessed
        
        Outputs:
        output: dict. Keys: [
            bboxes,  # Shape: (Nx4), 4 - [LeftTopX, LeftTopY, RightBottomX, RightBottomY]
            masks,   # Shape: (NxHxW), dtype: bool
            scores,  # Shape: (Nx1), dtype: float
        ]
        
        '''
        
        result = inference_detector(
            model=self.model,
            imgs=self.pre_process_(image),
        )
        
        return self.post_process_(result)
        
    def pre_process_(
        self,
        image: np.array,
    ) -> np.array:
    
        image[~self.mask] = 0
        
        return image
        
    def post_process_(
        self,
        mmcv_preds: tuple,
    ) -> dict:
    
        mmdet_bboxes, mmdet_masks = mmcv_preds
        out_bboxes, out_masks, out_scores = [], [], []
       
        for class_id in range(len(mmdet_bboxes)):
            for bbox, mask in zip(mmdet_bboxes[class_id], mmdet_masks[class_id]):
                x_l, y_l, x_r, y_r, sc = bbox
                if sc > self.treshold:
                    out_bboxes.append([x_l, y_l, x_r, y_r])
                    out_masks.append(mask)
                    out_scores.append(sc)
                
        return {
            'bboxes': np.array(out_bboxes),
            'masks': np.array(out_masks),
            'scores': np.array(out_scores),
        }         
