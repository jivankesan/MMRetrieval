# custom collate function to handle different video lengths:
import torch
import numpy as np


def custom_collate_fn(batch):
    videos, labels = zip(*batch)
    max_frames = max([video.shape[1] for video in videos])
    padded_videos = []
    
    for video in videos:
        padding = max_frames - video.shape[1]
        padded_video = np.pad(video, ((0, 0), (0, padding), (0, 0), (0, 0)), 'constant')
        padded_videos.append(padded_video)
    
    videos_tensor = torch.stack([torch.from_numpy(video) for video in padded_videos])
    labels_tensor = torch.stack(labels)
    
    return videos_tensor, labels_tensor