import torch
import torch.nn as nn
from utils.positions import classes, class_to_coding

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

def hemming_dist(output, target):
    predicted_class = classes[output]
    target_class = classes[target]
    
    pred_encoded = torch.tensor(class_to_coding[predicted_class])
    gt_encoded = torch.tensor(class_to_coding[target_class])
    dist = (pred_encoded != gt_encoded).sum()
    print(f"Predicted: {predicted_class}, Target: {target_class}, Distance: {dist}")
    return dist

class HammingLoss(nn.Module):
    def __init__(self):
        super(HammingLoss, self).__init__()

    def forward(self, output, target):
        loss = torch.tensor(0.0, device="cuda", requires_grad=True)    
        for i in range(len(output)):
            pred = output[i]
            predicted_class = classes[pred]
            tgt = target[i]
            target_class = classes[tgt]
            
            pred_encoded = torch.tensor(class_to_coding[predicted_class]).cuda()
            gt_encoded = torch.tensor(class_to_coding[target_class]).cuda()
            loss = loss + (pred_encoded != gt_encoded).sum().float()
        loss /= len(output)    
        return loss