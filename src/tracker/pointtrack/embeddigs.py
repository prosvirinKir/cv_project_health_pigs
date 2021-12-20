#!/usr/bin/python3

import cv2
import torch
import numpy as np

from tracker.pointtrack.utils import (
    get_model,
    preprocess,
)


class Embeddings:

    def __init__(
        self,
        kwargs: dict,
    ) -> None:
        
        main_model = kwargs['main']
        self.device = torch.device(kwargs['device'])
        
        self.model = get_model(
            kwargs=kwargs,
            device=self.device,
            classname=main_model,
        )
       
        self.model.eval()
        
        self.cat_emb = torch.tensor(kwargs['catEmb'], device=self.device)
        self.bg_num = int(kwargs['offsetEmb']['num_points'] / 3)
        self.fg_num = kwargs['offsetEmb']['num_points'] - self.bg_num
        
    def get_embeddings(
        self,
        image: np.array,
        masks: np.array,
    ) -> np.array:
    
        assert len(masks) != 0, 'Masks len is equal to zero.'
        
        tensor_image, tensor_masks = self.prepare_data(image, masks)
        
        points, xyxys = preprocess(
            image=tensor_image,
            masks=tensor_masks,
            cat_emb=self.cat_emb,
            bg_num=self.bg_num,
            fg_num=self.fg_num,
            device=self.device,
        )

        with torch.no_grad():
            embeds = self.model(
                points=points.unsqueeze(0),
                labels=None,
                xyxys=xyxys.unsqueeze(0),
                infer=True,
            ).squeeze(0).cpu().numpy()
            
        return embeds
    
    
    def prepare_data(
        self,
        image: np.array,
        masks: np.array,
    ) -> tuple:
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image = torch.from_numpy(image).to(self.device).float().div(255.0)
        masks = torch.from_numpy(masks).to(self.device)
        
        return (image, masks)
