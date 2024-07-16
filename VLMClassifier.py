from typing import List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.adapter import MLPAdapter, TransformerAdapter
from utils.train import train_adapter
from utils.test import test_adapter, inference_single_image
# from utils.classify import classify_zeroshot, classify_fewshotshot

# from BLIP.models.blip import blip_feature_extractor
from transformers import CLIPProcessor, CLIPModel

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class VLMClassifier:
    def __init__(self, 
                 device: torch.device, 
                 dtype: torch.dtype,
                 bs: int, 
                 adapter_image_type: str, 
                 adapter_descriptions_type: str, 
                 load_path: str, 
                 save_path: str, 
                 load_path_descriptions: str,
                 save_path_descriptions: str,
                 lr: float, 
                 weight_decay: float,
                 image_dir: str,
                 llava_path: str,
                 in_features: int
                 ) -> None:
        
        self.device = device
        self.dtype = dtype
        torch.set_default_dtype(self.dtype)

        self.bs = bs
        self.adapter_image_type = adapter_image_type
        self.adapter_descriptions_type = adapter_descriptions_type
        self.load_path = load_path
        self.save_path = save_path
        self.load_path_descriptions = load_path_descriptions
        self.save_path_descriptions = save_path_descriptions
        self.lr = lr
        self.weight_decay = weight_decay
        self.image_dir = image_dir
        self.warmup_epochs=3
        self.in_features = in_features
        self.dtype = dtype

        self.writer = SummaryWriter()

        self.llava_path = llava_path

        clip_model_name = 'openai/clip-vit-large-patch14-336' # 'openai/clip-vit-large-patch14-336', 'openai/clip-vit-base-patch16'
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)

        # model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'
        # self.blip_model = blip_feature_extractor(pretrained=model_url, image_size=800, vit='base').to(device)

        if self.adapter_image_type == 'mlp':
            self.adapter_image = MLPAdapter(in_features=self.in_features, hidden_features=256, dtype=self.dtype).to(device)
        elif self.adapter_image_type == 'transformer':
            self.adapter_image = TransformerAdapter(in_features=self.in_features, hidden_features=256, dtype=self.dtype).to(device)

        if self.adapter_descriptions_type == 'mlp':
            self.adapter_descriptions = MLPAdapter(in_features=self.in_features, hidden_features=256, dtype=self.dtype).to(device)
        elif self.adapter_descriptions_type == 'transformer':
            self.adapter_descriptions = TransformerAdapter(in_features=self.in_features, hidden_features=256, dtype=self.dtype).to(device)

        self.classes = ['back', 'bottom', 'bottomleftback', 'bottomleftfront', 'bottomrightback', 'bottomrightfront', 'front', 
                        'left', 'right', 'top', 'topleftback', 'topleftfront', 'toprightback', 'toprightfront']

        self.questions = self._prepare_prompt()
        self.metrics = self._reset_metrics()

        self.optimizer_image = optim.Adam(self.adapter_image.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer_descriptions = optim.Adam(self.adapter_descriptions.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = HammingLoss() #nn.CrossEntropyLoss()

        self.scheduler_image = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_image, T_max=3, eta_min=1e-6)
        self.scheduler_descriptions = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_descriptions, T_max=3, eta_min=1e-6)
        

    def _reset_metrics(self) -> Dict[str, Dict[str, List[int]]]:
        self.metrics = {'train': {'gts': [], 'preds': []}, 'val': {'gts': [], 'preds': []}, 'test': {'gts': [], 'preds': []}}
        return self.metrics
    
    def _prepare_prompt(self) -> List[str]:
        questions = [f"a remote observed from {CLASS_NAME}" for CLASS_NAME in self.classes]
        return questions
    
    def get_all_parameters(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, nn.Module):
                for name, param in attr.named_parameters():
                    print(f"Model: {attr_name}, Parameter: {name}, Dtype: {param.dtype}, Device: {param.device}")

class HammingLoss(nn.Module):
    def __init__(self):
        super(HammingLoss, self).__init__()

        '''
        0,      1,       2
        top,    bottom,  null 
        left,   right,   null 
        front,  back,    null
        '''
        
        self.class_to_coding = {
            "topleftfront": [0, 0, 0],
            "topleftback": [0, 0, 1],
            "topleft": [0, 0, 2],
            "toprightfront": [0, 1, 0],
            "toprightback": [0, 1, 1],
            "topright": [0, 1, 2],
            "topfront": [0, 2, 0],
            "topback": [0, 2, 1],
            "top": [0, 2, 2],
            "bottomleftfront": [1, 0, 0],
            "bottomleftback": [1, 0, 1],
            "bottomleft": [1, 0, 2],
            "bottomrightfront": [1, 1, 0],
            "bottomrightback": [1, 1, 1],
            "bottomright": [1, 1, 2],
            "bottomfront": [1, 2, 0],
            "bottomback": [1, 2, 1],            
            "bottom": [1, 2, 2],
            "leftfront": [2, 0, 0],
            "leftback": [2, 0, 1],
            "left": [2, 0, 2],
            "rightfront": [2, 1, 0],
            "rightback": [2, 1, 1],
            "right": [2, 1, 2],
            "front": [2, 2, 0],
            "back": [2, 2, 1],
            "null": [2, 2, 2]
        }

        self.classes = list(self.class_to_coding.keys())

    def forward(self, output, target):
        loss = torch.tensor(0.0, device="cuda", requires_grad=True)    
        for i in range(len(output)):
            pred = output[i]
            predicted_class = self.classes[pred]
            tgt = target[i]
            target_class = self.classes[tgt]
            
            pred_encoded = torch.tensor(self.class_to_coding[predicted_class]).cuda()
            gt_encoded = torch.tensor(self.class_to_coding[target_class]).cuda()
            loss = loss + (pred_encoded != gt_encoded).sum().float()
        loss /= len(output)    
        return loss

if __name__ == '__main__':

    hparams = {
        'save_path': 'ckpts/adapter_image.pth',
        'save_path_descriptions': 'ckpts/adapter_descriptions.pth', 

        'load_path': 'ckpts/adapter_image.pth',
        'load_path_descriptions': 'ckpts/adapter_descriptions.pth',

        'device': torch.device("cuda"),
        'dtype': torch.float32,
        'image_dir': 'data/remote60',
        'in_features': 768, #512 for clip base, 768 for clip large
        'llava_path': "llava-hf/llava-1.5-7b-hf",

        'adapter_image_type': 'mlp', # 'mlp', 'transformer'
        'adapter_descriptions_type': 'transformer', # 'mlp', 'transformer'
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'bs': 4, #16
    }

    classifier = VLMClassifier(**hparams)

    hparams = {
        'model_class': classifier, 
        'epochs': 100,
        'train_descriptions': "descriptions/train_descriptions.json",
        'val_descriptions': "descriptions/val_descriptions.json",
        'lam': 1
    }
    train_adapter(**hparams)
    
    # test_adapter(model_class=classifier, split='test', plot=True, fusion=False, lam=0.1)
    # inference_single_image(model_class=classifier, image_path='data/remote14/train/remote-comfee/BottomRightBack.png', plot=True)

