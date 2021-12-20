#!/usr/bin/python3

import torch
import munkres
import collections
import pycocotools.mask as maskUtils
import numpy as np
from scipy.spatial.distance import cdist, euclidean

'''
t - frame count
id - position in input data
track_id - assigned id
class_id - class id
mask - instance mask in np.array style
embed - embedding in np.array style
mean - center of the object in np.array style
'''

TrackElement = collections.namedtuple(
    'TrackElement',
    ['t', 'id', 'track_id', 'class_id', 'mask', 'embed', 'mean'],
)

class TrackHelper:

    def __init__(
        self,
        keep_alive: int,
        mask_iou: bool,
        euclidean_scale: float,
        euclidean_offset: float,
        association_threshold: float,
        means_threshold: float,
        mask_iou_scale: float,
        class_id: int,
    ) -> None:

        self.keep_alive = keep_alive
        self.mask_iou = mask_iou
        self.euclidean_scale = euclidean_scale
        self.euclidean_offset = euclidean_offset
        self.association_threshold = association_threshold
        self.means_threshold = means_threshold
        self.mask_iou_scale = mask_iou_scale
        self.class_id = class_id

        self.munkres_obj = munkres.Munkres()
        self.active_tracks = []
        self.next_inst_id = None

    def update_active_track(
        self,
        frame_count: int
    ) -> None:

        self.active_tracks = [
            track for track in self.active_tracks
            if track.t >= frame_count - self.keep_alive
        ]

    def assign(
        self,
        frame_count: int,
        embeds: np.array,
        masks: np.array,
    ) -> list:
    
        ''' Main step of Pointtrack assigner
        
        Inputs:
        frame_count: int. Like abstract time,
        embeds: np.array. Shape: (Nx128), dtype: float
        masks: np.array. Shape: (NxHxW), dtype: bool
        
        Ouputs:
        result: list. List with assigned indexes for each instance in the same order as input data
        '''

        if self.next_inst_id is None:
            self.next_inst_id = 1
        else:
            self.update_active_track(frame_count)
        
        n = len(embeds)
        result = [0]*n
        
        if n < 1:
            return result
        
        means = [
            np.stack(np.nonzero(mask)).mean(axis=1)
            for mask in masks
        ]
        masks = [
            maskUtils.encode(np.asfortranarray(v.astype(np.uint8)))
            for v in masks
        ]
        
        if len(self.active_tracks) == 0:
            for i in range(n):
                self.active_tracks.append(
                    TrackElement(
                        t=frame_count,
                        id=i,
                        mask=masks[i],
                        class_id=self.class_id,
                        track_id=self.next_inst_id,
                        embed=embeds[i],
                        mean=means[i],
                    )
                )
                result[i] = self.next_inst_id
                self.next_inst_id += 1

            return result
        
        # compare inst by inst.
        # only compare with previous embeds, not including embeds of this frame
        last_reids = np.concatenate(
            [
                el.embed[np.newaxis]
                for el in self.active_tracks
            ], axis=0,
        )
        
        # cost matrix
        asso_sim = np.zeros(
            shape=(n, len(self.active_tracks)),
            dtype=np.float32,
        )
        
        # array for assigned instances
        detections_assigned = np.zeros(
            shape=len(embeds),
            dtype=bool,
        )
        
        # step 1. Use distance between embeddings
        asso_sim += self.euclidean_scale * (
              self.euclidean_offset - cdist(embeds, last_reids)
        )
        
        # step 2. Use distance between masks (IoU)
        if self.mask_iou:
            asso_sim += self.mask_iou_scale * maskUtils.iou(
                masks,
                [v.mask for v in self.active_tracks],
                np.zeros(len(self.active_tracks)),
            )
        
        cost_matrix = munkres.make_cost_matrix(asso_sim)
        
        for row, column in np.argwhere(asso_sim <= self.association_threshold):
            cost_matrix[row][column] = 1e9
            
        for row, column in self.munkres_obj.compute(cost_matrix):
        
            # the instance was the same
            if euclidean(self.active_tracks[column].mean, means[row]) < self.means_threshold \
             and cost_matrix[row][column] != 1e9:
                current_inst = TrackElement(
                    t=frame_count,
                    id=row,
                    mask=masks[row],
                    class_id=self.class_id,
                    track_id=self.active_tracks[column].track_id,
                    embed=embeds[row],
                    mean=means[row],
                )
                self.active_tracks[column] = current_inst
                result[row] = self.active_tracks[column].track_id
            
            # it is a new instance
            else:
                current_inst = TrackElement(
                    t=frame_count,
                    id=row,
                    mask=masks[row],
                    class_id=self.class_id,
                    track_id=self.next_inst_id,
                    embed=embeds[row],
                    mean=means[row],
                )
                self.active_tracks.append(current_inst)
                result[row] = self.next_inst_id
                self.next_inst_id += 1

            detections_assigned[row] = True

        # new track id for unassigned instances
        for i in np.nonzero(detections_assigned == False)[0]:
            current_inst = TrackElement(
                t=frame_count,
                id=i,
                mask=masks[i],
                class_id=self.class_id,
                track_id=self.next_inst_id,
                embed=embeds[i],
                mean=means[i],
            )
            self.active_tracks.append(current_inst)
            result[i] = self.next_inst_id
            self.next_inst_id += 1

        return result
