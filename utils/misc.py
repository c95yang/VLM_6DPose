import torch
import torch.nn as nn
from .positions import classes, class_to_coding
import numpy as np

import psutil
import os
import tracemalloc

# Start memory trace
# tracemalloc.start()

def interpolate_color(dist, max_dist=3):
    t = dist / max_dist
    red = np.array([255, 0, 0])
    green = np.array([0, 255, 0])
    
    color = (1 - t) * green + t * red
    return tuple(color / 255)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"RSS (Resident Set Size): {mem_info.rss / 1024 ** 2:.2f} MB")
    print(f"VMS (Virtual Memory Size): {mem_info.vms / 1024 ** 2:.2f} MB")
    
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print("[ Top 5 Memory Consumers ]")
    for stat in top_stats[:5]:
        print(stat)

def parse_output(output: str) -> str:
    return output.split("ASSISTANT:")[-1].strip()

def calculate_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images_count = 0
    for images, _, _ in loader:
        batch_images_count = images.size(0)
        images = images.view(batch_images_count, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_images_count
    mean /= total_images_count
    std /= total_images_count
    print(f"Mean: {mean}, Std: {std}, loader: {loader}")
    return mean, std

def hamming_dist(gt, pred):
    pred_class = classes[pred]
    gt_class = classes[gt]
    pred_encoded = torch.tensor(class_to_coding[pred_class])
    gt_encoded = torch.tensor(class_to_coding[gt_class])
    dist = (pred_encoded != gt_encoded).sum()
    return dist

class HammingLoss(nn.Module):
    def __init__(self):
        super(HammingLoss, self).__init__()

    def forward(self, output, target):
        loss = torch.tensor(0.0, device="cuda", requires_grad=True)    
        for i in range(len(output)):
            pred = output[i]
            gt = target[i]
            loss += hamming_dist(gt, pred).float()
        loss /= len(output)    
        return loss