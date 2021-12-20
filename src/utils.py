#!/usr/bin/python3

import cv2
import numpy as np

map_activity = {
    'sedentary': (0, 0, 255),
    'neutral': (0, 255, 255),
    'active': (0, 252, 124),
}

class ColorGenerator:

    def __init__(
        self,
        n: int = 30,
    ) -> None:
    
        self.n = n
        self.colors = np.random.randint(
            low=0,
            high=255,
            size=(self.n, 3),
        )

    def get_color(
        self,
        track_id: int,
    ) -> np.array:

        return self.colors[track_id % self.n]

def visualize(
    image: np.array,
    masks: np.array,
    indexes: list,
    activities: list,
    colors: ColorGenerator,
) -> np.array:

    for mask, track_id, activity in zip(masks, indexes, activities):
    
        image[mask] = image[mask] * 0.5 + colors.get_color(track_id) * 0.5
        
        # find bbox
        x, y = np.nonzero(mask)
        x_l, x_r = int(np.min(x).item()), int(np.max(x).item())
        y_l, y_r = int(np.min(y).item()), int(np.max(y).item())
        
        # draw bbox
        cv2.rectangle(
            img=image,
            pt1=(y_l, x_l),
            pt2=(y_r, x_r),
            color=colors.get_color(track_id).tolist(),
            thickness=3,
        )
        
        color_activity = map_activity[activity[1]]
       
        # draw track_id
        x_c = int(np.mean(x).item())
        y_c = int(np.mean(y).item())

        cv2.putText(
            img=image,
            text=f'{track_id}',
            org=(y_c, x_c),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=color_activity,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
       
        # draw activity
        x_new = np.max([0, x_c - 40])
        cv2.putText(
            img=image,
            text=activity[0],
            org=(y_c, x_new),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=2,
            color=color_activity,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
        
    x_corner, y_corner = 250, 250
    cv2.putText(
        img=image,
        text=f'Pigs:{len(indexes)}',
        org=(y_corner, x_corner),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=2,
        color=(0, 0, 255),
        thickness=3,
        lineType=cv2.LINE_AA,
    )
    
    return image

def crop(
    bbox: list, 
    mask: np.array,
) -> np.array:

    '''Crop submask from mask image by bbox'''

    bbox[0::2] = np.clip(bbox[0::2], 0, mask.shape[0])
    bbox[1::2] = np.clip(bbox[1::2], 0, mask.shape[1])
    x1, y1, x2, y2 = map(int, bbox)
    w = np.maximum(x2 - x1, 1)
    h = np.maximum(y2 - y1, 1)
    cropped_mask = mask[y1:y1 + h, x1:x1 + w]

    return cropped_mask
