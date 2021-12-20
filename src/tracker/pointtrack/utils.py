#!/usr/bin/python3

import torch
import numpy as np
from collections import OrderedDict

from tracker.pointtrack.models.branchedERFNet import TrackerOffsetEmb


def get_model(
    kwargs: dict,
    device: torch.device,
    classname: str,
) -> TrackerOffsetEmb:

    model = TrackerOffsetEmb(device, **kwargs['offsetEmb']).to(device)

    state = torch.load(
        f=kwargs['model'][classname],
        map_location='cpu',
    )
    if 'model_state_dict' in state:
        state = state['model_state_dict']

    # remove 'module.' of dataparallel
    new_state = OrderedDict()
    for k in state:
        new_state[k[7:]] = state[k]

    model.load_state_dict(new_state)

    return model

def crop(
    mask: torch.tensor,
    field: torch.tensor,
    image: torch.tensor,
    minMaxU: tuple,
    minMaxV: tuple,
) -> tuple:

    vMax, uMax, _ = image.size()
    u0, u1 = minMaxU
    v0, v1 = minMaxV

    vd = v1.add_(1).sub(v0).mul_(0.2)
    ud = u1.add_(1).sub(u0).mul_(0.2)
    
    v0 = v0.sub_(vd).clamp_(min=0).int()
    v1 = v1.add_(vd).clamp_(max=vMax-1).int()
    u0 = u0.sub_(ud).clamp_(min=0).int()
    u1 = u1.add_(ud).clamp_(max=uMax-1).int()
             
    field[mask] = 1
    crop_field = field[v0:v1, u0:u1].clone()
    field[mask] = 2

    crop_mask = mask[v0:v1, u0:u1].clone()
    crop_image = image[v0:v1, u0:u1].clone()
    
    return crop_mask, crop_image, crop_field

def preprocess(
    image: torch.tensor,
    masks: torch.tensor,
    cat_emb: torch.tensor,
    bg_num: int,
    fg_num: int,
    device: torch.device,
) -> tuple:

    n = len(masks)
    vMax, uMax, _ = image.size()
    points, xyxys = [0]*n, [0]*n
    field = masks.sum(dim=0, dtype=torch.bool).long().mul_(2)

    for i in range(n):
        vs, us = masks[i].nonzero(as_tuple=True)

        v0, v1 = vs.min().float(), vs.max().float()
        u0, u1 = us.min().float(), us.max().float()

        xyxys[i] = torch.stack([u0.div(uMax), v0.div(vMax), u1.div(uMax), v1.div(vMax)])

        crop_mask, crop_image, crop_field = crop(
            masks[i], field, image, (u0, u1), (v0, v1))

        vs, us = crop_mask.nonzero(as_tuple=True)

        vs, us = vs.float(), us.float()
        vc, uc = vs.mean(), us.mean()

        vs.sub_(vc).div_(128.0).unsqueeze_(-1)
        us.sub_(uc).div_(128.0).unsqueeze_(-1)

        pointUVs = torch.cat([crop_image[crop_mask], vs, us], dim=1)
        points_fg = torch.cat([
            pointUVs[torch.randint(high=pointUVs.size(0), size=(fg_num, ))].unsqueeze(0),
            torch.zeros((1, fg_num, 3), dtype=torch.float, device=device)
        ], dim=-1)

        crop_mask = ~crop_mask

        if crop_mask.sum() == 0:
            points_bg = torch.zeros((1, bg_num, 8), dtype=torch.float, device=device)
        else:
            vs, us = crop_mask.nonzero(as_tuple=True)

            vs = vs.float().sub_(vc).div_(128.0).unsqueeze_(-1)
            us = us.float().sub_(uc).div_(128.0).unsqueeze_(-1)

            pointUVs = torch.cat([
                crop_image[crop_mask], vs, us,
                torch.index_select(cat_emb, 0, crop_field[crop_mask])], dim=1)
            points_bg = pointUVs[torch.randint(high=pointUVs.size(0), size=(bg_num, ))].unsqueeze(0)

        points[i] = torch.cat([points_fg, points_bg], dim=1)

    points = torch.cat(points, dim=0)
    xyxys = torch.stack(xyxys)

    return points, xyxys
