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

class CLIPClassifier:
    def __init__(self, 
                 device: torch.device, 
                 bs: int, 
                 model_name: str, 
                 adapter_type: str, 
                 load_path: str, 
                 save_path: str, 
                 lr, 
                 weight_decay,
                 image_dir,
                 ) -> None:
        
        self.device = device
        self.bs = bs
        self.model_name = model_name
        self.adapter_type = adapter_type
        self.load_path = load_path
        self.save_path = save_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.image_dir = image_dir

        self.writer = SummaryWriter()

        self.model = CLIPModel.from_pretrained(self.model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        if self.adapter_type == 'mlp':
            self.adapter = MLPAdapter().to(device)
        elif self.adapter_type == 'transformer':
            self.adapter = TransformerAdapter().to(device)
        elif self.adapter_type == 'mamba':
            self.adapter = MambaAdapter().to(device)

        self.classes = ['back', 'bottom', 'bottomleftback', 'bottomleftfront', 'bottomrightback', 'bottomrightfront', 'front', 
                        'left', 'right', 'top', 'topleftback', 'topleftfront', 'toprightback', 'toprightfront']
        self.questions = self._prepare_prompt()
        self.metrics = self._reset_metrics()

        self.optimizer = optim.Adam(self.adapter.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        self.warmup_epochs=3
        self.scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=3, eta_min=1e-6)
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=self.warmup_epochs, after_scheduler=self.scheduler_cosine)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

    def _reset_metrics(self) -> Dict[str, Dict[str, List[int]]]:
        self.metrics = {'train': {'gts': [], 'preds': []}, 'val': {'gts': [], 'preds': []}, 'test': {'gts': [], 'preds': []}}
        return self.metrics
    
    def _prepare_prompt(self) -> List[str]:
        questions = [f"A remote control device observed from {CLASS_NAME} direction." for CLASS_NAME in self.classes]
        #questions.extend([f"A remote observed from {CLASS_NAME} direction." for CLASS_NAME in self.classes])
        #questions.extend([f"A photo of remote taken from {CLASS_NAME} direction." for CLASS_NAME in self.classes])
        return questions

if __name__ == '__main__':
    # classify_zeroshot(model_class=classifier, split='train' )
    # classify_fewshotshot(model_class=classifier, split='train')

    hparams = {
        'model_name': 'openai/clip-vit-base-patch16', # 'openai/clip-vit-large-patch14-336', 'openai/clip-vit-base-patch16'
        'adapter_type': 'mamba', # 'mlp', 'transformer', 'mamba'
        'save_path': 'ckpts/adapter.pth',
        'load_path': 'ckpts/adapter.pth',
        'device': torch.device("cuda"),
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'bs': 32,
        'image_dir': 'data/remote14',
    }

    classifier = CLIPClassifier(**hparams)

    train_adapter(model_class=classifier, epochs=300, pureclip=False)
    # test_adapter(model_class=classifier, split='val')
