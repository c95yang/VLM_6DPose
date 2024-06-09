import torch.nn as nn
from mamba_ssm import Mamba

class MLPAdapter(nn.Module):
    def __init__(self, in_features=512, hidden_features=512, dropout=0.1):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.dropout = dropout

        self.layers = nn.Sequential(

            # nn.Dropout(self.dropout),
            # nn.BatchNorm1d(self.in_features),
            # nn.Linear(in_features=self.in_features, out_features=self.hidden_features),
            # nn.GELU(),

            nn.BatchNorm1d(self.in_features),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=self.in_features, out_features=self.hidden_features),
            nn.GELU(),
            nn.BatchNorm1d(self.hidden_features),
            nn.Dropout(self.dropout)
            )
        
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, feat):
        img_feat = self.layers(feat)
        return img_feat


class TransformerAdapter(nn.Module):
    def __init__(self, in_features=512, hidden_features=512, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.nhead = 8
        self.dropout = dropout
        self.num_layers = 2
         
        self.input_projection = nn.Sequential(
            nn.BatchNorm1d(self.in_features),
            nn.Dropout(self.dropout),
            # nn.Linear(in_features=self.in_features, out_features=self.hidden_features),
            # nn.GELU()
        )

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_features, nhead=self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.output_projection = nn.Sequential(
            nn.BatchNorm1d(self.hidden_features),
            nn.Dropout(self.dropout),
            # nn.Linear(in_features=self.hidden_features, out_features=self.in_features),
            # nn.GELU()
        )

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        out = self.input_projection(x).unsqueeze(0)
        out = self.transformer_encoder(out).squeeze(0)
        out = self.output_projection(out)
        return out    

class MambaAdapter(nn.Module):
    def __init__(self, in_features=512, hidden_features=512, dropout=0.2):
        super().__init__()
        self.in_features = in_features
        self.dropout = dropout

        self.mamba = nn.Sequential(
            Mamba(d_model=512),
        )

        self.projection = nn.Sequential(
            nn.BatchNorm1d(self.in_features),
            nn.Dropout(self.dropout),
            # nn.Linear(in_features=self.in_features, out_features=self.in_features),
            # nn.GELU(),
        )

        for param in self.parameters():
            param.requires_grad = True

    def forward(self, feat):
        out = self.projection(feat)

        out = out.unsqueeze(1)
        out = self.mamba(out)
        out = out.squeeze(1)

        # out = self.projection(out)

        # out = out.unsqueeze(1)
        # out = self.mamba(out)
        # out = out.squeeze(1)

        out = self.projection(out)
        return out
