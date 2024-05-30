import numpy as np
import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self):
        super().__init__()

        # self.adapter_ratio = 1
        self.dropout = 0.075

        self.layers = nn.Sequential(

            # nn.Dropout(self.dropout),
            # nn.BatchNorm1d(self.in_features),
            # nn.Linear(in_features=self.in_features, out_features=self.hidden_features),
            # nn.GELU(),

            nn.BatchNorm1d(512),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=512, out_features=512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(self.dropout)
            )
        
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, feat):
        img_feat = self.layers(feat) #([266, 512])
        # img_feat = img_feat * self.adapter_ratio + feat * (1 - self.adapter_ratio)
        return img_feat
