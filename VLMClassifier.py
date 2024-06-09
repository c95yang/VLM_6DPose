from typing import List, Dict
import torch
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

from models.adapter import MLPAdapter, TransformerAdapter, MambaAdapter
from utils.train import train_adapter
from utils.test import test_adapter
from utils.classify import classify_zeroshot, classify_fewshotshot

class VLMClassifier:
    def __init__(self, 
                 device: torch.device, 
                 bs: int, 
                 model_name: str, 
                 adapter_image_type: str, 
                 adapter_descriptions_type: str, 
                 load_path: str, 
                 save_path: str, 
                 load_path_descriptions: str,
                 save_path_descriptions: str,
                 lr, 
                 weight_decay,
                 image_dir,
                 ) -> None:
        
        self.device = device
        self.bs = bs
        self.model_name = model_name
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

        self.writer = SummaryWriter()

        self.model = CLIPModel.from_pretrained(self.model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        if self.adapter_image_type == 'mlp':
            self.adapter_image = MLPAdapter().to(device)
        elif self.adapter_image_type == 'transformer':
            self.adapter_image = TransformerAdapter().to(device)
        elif self.adapter_image_type == 'mamba':
            self.adapter_image = MambaAdapter().to(device)

        if self.adapter_descriptions_type == 'mlp':
            self.adapter_descriptions = MLPAdapter().to(device)
        elif self.adapter_descriptions_type == 'transformer':
            self.adapter_descriptions = TransformerAdapter().to(device)
        elif self.adapter_descriptions_type == 'mamba':
            self.adapter_descriptions = MambaAdapter().to(device)

        self.classes = ['back', 'bottom', 'bottomleftback', 'bottomleftfront', 'bottomrightback', 'bottomrightfront', 'front', 
                        'left', 'right', 'top', 'topleftback', 'topleftfront', 'toprightback', 'toprightfront']
        self.questions = self._prepare_prompt()
        self.metrics = self._reset_metrics()

        self.optimizer_image = optim.Adam(self.adapter_image.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer_descriptions = optim.Adam(self.adapter_descriptions.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        self.scheduler_cosine_image = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_image, T_max=3, eta_min=1e-6)
        self.scheduler_image = GradualWarmupScheduler(self.optimizer_image, multiplier=1, total_epoch=self.warmup_epochs, after_scheduler=self.scheduler_cosine_image)
        self.scheduler_cosine_descriptions = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_descriptions, T_max=3, eta_min=1e-6)
        self.scheduler_descriptions = GradualWarmupScheduler(self.optimizer_descriptions, multiplier=1, total_epoch=self.warmup_epochs, after_scheduler=self.scheduler_cosine_descriptions)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

    def _reset_metrics(self) -> Dict[str, Dict[str, List[int]]]:
        self.metrics = {'train': {'gts': [], 'preds': []}, 'val': {'gts': [], 'preds': []}, 'test': {'gts': [], 'preds': []}}
        return self.metrics
    
    def _prepare_prompt(self) -> List[str]:
        questions = [f"A remote control device observed from {CLASS_NAME} direction." for CLASS_NAME in self.classes]
        #questions.extend([f"A remote observed from {CLASS_NAME} direction." for CLASS_NAME in self.classes])
        return questions

if __name__ == '__main__':
    # classify_zeroshot(model_class=classifier, split='train' )
    # classify_fewshotshot(model_class=classifier, split='train')

    hparams = {
        'save_path': 'ckpts/adapter_image.pth',
        'save_path_descriptions': 'ckpts/adapter_descriptions.pth', 
        'load_path': 'ckpts/adapter_image.pth',
        'load_path_descriptions': 'ckpts/adapter_descriptions.pth',
        'device': torch.device("cuda"),
        'image_dir': 'data/remote14',
        'model_name': 'openai/clip-vit-base-patch16', # 'openai/clip-vit-large-patch14-336', 'openai/clip-vit-base-patch16'
        'adapter_image_type': 'transformer', # 'mlp', 'transformer', 'mamba'
        'adapter_descriptions_type': 'transformer', # 'mlp', 'transformer', 'mamba'
        'lr': 1e-5,
        'weight_decay': 1e-4,
        'bs': 16,
    }

    classifier = VLMClassifier(**hparams)

    train_adapter(model_class=classifier, epochs=300)
    # test_adapter(model_class=classifier, split='test')
