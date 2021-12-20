#!/usr/bin/python3

import os
import cv2
import yaml
import numpy as np

def get_way(x_1, y_1, x_2, y_2):
    dx = x_2 - x_1
    dy = y_2 - y_1
    d_way = np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)
    return d_way

def dice(im1, im2):
    return 2 * np.logical_and(im1, im2).sum() / (im1.sum() + im2.sum())
   

class Activity:

    def __init__(
        self,
        config: str,
    ) -> None:
        
        assert os.path.exists(config), f'Config file {config} does not exist!'
        
        with open(config, 'r') as file:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)
        
        self.treshold = kwargs['treshold']
        self.water_l = np.load(kwargs['water_l'])
        self.water_r = np.load(kwargs['water_r'])
        self.feed = np.load(kwargs['feed'])
        self.low = kwargs['low']
        self.high = kwargs['high']
        
        self.dictionary = {}
        self.ways = {}
    
    def get_activity(
        self,
        masks: np.array,
        indexes: list,
    ) -> list:
    
        answer = []
        
        for index, mask in zip(indexes, masks):
        
            if index in self.dictionary:
            
                self.dictionary[index].append(mask)
                
                mask_current = self.dictionary[index][-1]
                mask_past = self.dictionary[index][-2]
                
                x_1, y_1 = np.nonzero(mask_past)
                x_1 = int(np.mean(x_1).item())
                y_1 = int(np.mean(y_1).item())

                x_2, y_2 = np.nonzero(mask_current)
                x_2 = int(np.mean(x_2).item())
                y_2 = int(np.mean(y_2).item())
                
                d_way = get_way(x_1, y_1, x_2, y_2)
                
                self.ways[index].append(d_way)
                
                conditions, verdict = self.statement(mask_current, self.ways[index], d_way)
                
            else:
                conditions, verdict = 'in place', 'neutral'
                self.ways[index] = [0]
                self.dictionary[index] = [mask]
                
            answer.append((conditions, verdict))
            
        return answer

    def statement(
        self,
        mask: np.array,
        way: list,
        d_way: float,
    ) -> tuple:
    
        verdict = 'neutral'
        
        if d_way > self.high:
            conditions = 'walk'
        else:
            conditions = 'chill'
            
        if dice(mask, self.feed) > self.treshold:
            conditions = 'eat'
        elif (dice(mask, self.water_l) > self.treshold) or \
             (dice(mask, self.water_r) > self.treshold):
            conditions = 'drink'
            
        if conditions != 'chill':
            verdict = 'active'
        else:
            if np.mean(way) < self.low:
                verdict = 'sedentary'
            elif self.low <= np.mean(way) <= 2 * self.high:
                verdict = 'neutral'
            else:
                verdict = 'active'
                
        return conditions, verdict
    
    def reset(
        self
    ) -> None:
            
        ''' Reset tracker for new sequence of data'''
        self.dictionary = {}
        self.ways = {}
