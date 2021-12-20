#!/usr/bin/python3

import os
import cv2
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm

from saver import savePredict

from detector.main import Detector
from tracker.pointtrack.main import Tracker
from activity import Activity

from utils import (
    ColorGenerator,
    visualize,
    crop,
)

RESIZE_X, RESIZE_Y = 1700, 1700

class PigTrack:

    def __init__(
        self,
        config: str='../configs/main.yaml',
    ) -> None:
    
        assert os.path.exists(config), f'Config file {config} does not exist!'
        
        with open(config, 'r') as file:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)

        self.detector = Detector(kwargs['detector_cfg_path'])
        self.tracker = Tracker(kwargs['tracker_cfg_path'])
        self.activity = Activity(kwargs['activity_cfg_path'])
        
        self.colors = ColorGenerator(30)
        
    def track_on_video(
        self,
        video_path_input: str,
        video_path_output: str,
    ) -> list:
    
        cap = cv2.VideoCapture(video_path_input)
        out = cv2.VideoWriter(
            filename=video_path_output,
            fourcc=cv2.VideoWriter_fourcc(*'XVID'),
            fps=10,
            frameSize=(RESIZE_X, RESIZE_Y),
            isColor=True,
        )
        
        assert cap.isOpened(), 'Error opening video stream or file'
        #ret, frame = cap.read()

        bboxes_per_frame, masks_per_frame = [], []
        
        while cap.isOpened():
        
            ret, frame = cap.read()
            
            if ret:
               
                detections = self.detector.detect(frame)
                indexes = self.tracker.track(
                    image=frame,
                    masks=detections['masks'],
                )
                activities = self.activity.get_activity(
                    masks=detections['masks'],
                    indexes=indexes,
                )

                #save video
                frame = visualize(
                    image=frame,
                    masks=detections['masks'],
                    indexes=indexes,
                    activities=activities,
                    colors=self.colors,
                )
                
                out.write(frame)

                #print("----")
                #print("Get ", len(indexes), " pigs")
                #print(indexes)
                #for i in range(len(indexes)):
                #    print(detections["bboxes"][i])
                if len(indexes) != 0:
                    sorted_bboxes = [[] for i in range(max(indexes))]
                    sorted_masks = [[] for i in range(max(indexes))]

                    for i in range(len(indexes)):
                        sorted_bboxes[indexes[i] - 1] = detections["bboxes"][i].copy()
                        sorted_masks[indexes[i] - 1] = detections["masks"][i].copy()

                    detections = {
                        'bboxes': sorted_bboxes,
                        'masks': sorted_masks
                    }
                else:
                    detections = {
                        'bboxes': [],
                        'masks': [],
                    }
               
                #print("After sorting")
                #for i in range(max(indexes)):
                #    if i+1 in indexes:
                #        print(i + 1, "==", detections["bboxes"][i])
                #    else:
                #        print("..", "==", detections["bboxes"][i])

                #for hdf5 file
                cropped_and_binarized_masks = [
                    crop(bbox, detections['masks'][i]).astype(np.uint8) if len(bbox) else []
                    for i, bbox in enumerate(detections['bboxes'])
                ]

                bboxes_per_frame.append(detections['bboxes'])
                masks_per_frame.append(cropped_and_binarized_masks)
                
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        
        cap.release()
        out.release()
        
        return bboxes_per_frame, masks_per_frame
        
        
def main(
    args,
) -> None:
    
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    os.makedirs(args.output_folder, exist_ok=True)
        
    pig_track = PigTrack()

    for video_path in tqdm(sorted(os.listdir(args.input_folder))):
        filename, file_extension = os.path.splitext(video_path)
        boxes_per_frame, masks_per_frame = pig_track.track_on_video(
            video_path_input=os.path.join(args.input_folder, video_path),
            video_path_output=os.path.join(args.output_folder, f'{filename}_annotated{file_extension}'),
        )
        pig_track.tracker.reset()
        pig_track.activity.reset()
        
        savePredict(
            Path=args.output_folder,
            Name=f'{filename}_annotation.hdf5',
            boxs=boxes_per_frame,
            masks=masks_per_frame,
        )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_folder',
        type=str,
        help='path to folder with videos',
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        help='path to folder with results (videos with visualization and hdf5 files")',
    )
    args = parser.parse_args()
    
    main(args)
