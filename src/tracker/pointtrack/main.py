#!/usr/bin/python3

import os
import torch
import yaml
import numpy as np

from tracker.pointtrack.association import TrackHelper
from tracker.pointtrack.embeddigs import Embeddings
from tracker.pointtrack.utils import (
    get_model,
    preprocess,
)

class Tracker:

    def __init__(
        self,
        config: str,
    ) -> None:
    
        assert os.path.exists(config), f'Config file {config} does not exist!'
        
        with open(config, 'r') as file:
            self._kwargs = yaml.load(file, Loader=yaml.FullLoader)
            
        self._main_class = self._kwargs['main']
            
        self.assigner = TrackHelper(**self._kwargs['trackHelper'][self._main_class])
        self.embedder = Embeddings(self._kwargs)
        
        self.frame_count = -1
        
    def track(
        self,
        image: np.array,
        masks: np.array,
    ) -> list:
    
        ''' Main step of Pointtrack tracker
        
        Inputs:
        image: np.array. Shape (HxWx3), BGR channel order, not preprocessed
        masks: np.array. Shape (NxHxW), dtype bool
        
        Ouputs:
        result: list. List with assigned indexes for each instance in the same order as input data
        '''
        
        self.frame_count += 1
        
        if len(masks) == 0:
            return []
        
        embeds = self.embedder.get_embeddings(
            image=image,
            masks=masks,
        )
        
        indexes = self.assigner.assign(
            frame_count=self.frame_count,
            embeds=embeds,
            masks=masks,
        )
        
        return indexes

    def reset(
        self
    ) -> None:

        ''' Reset tracker for new sequence of data'''
        self.assigner = TrackHelper(**self._kwargs['trackHelper'][self._main_class])
