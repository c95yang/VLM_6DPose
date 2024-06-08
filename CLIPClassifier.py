from typing import List, Dict
import torch
from torch.utils.data import DataLoader
from datasets import Remote14
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import torch.optim as optim
from adapter import MLPAdapter, TransformerAdapter, MambaAdapter
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler
from train import train_adapter
from test import test_adapter
from classify import classify_zeroshot, classify_fewshotshot

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
        self.save_path = save_path
        self.load_path = load_path
        self.adapter_type = adapter_type
        self.device = device
        self.model_name = model_name
        self.writer = SummaryWriter()
        self.lr = lr
        self.weight_decay = weight_decay
        self.bs = bs
        self.image_dir = image_dir

        self.model = CLIPModel.from_pretrained(self.model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

        if self.adapter_type == 'mlp':
            self.adapter = MLPAdapter().to(device)
        elif self.adapter_type == 'transformer':
            self.adapter = TransformerAdapter().to(device)
        elif self.adapter_type == 'mamba':
            self.adapter = MambaAdapter().to(device)

        self.train_dataset = Remote14(root_dir=self.image_dir, is_train=True)
        self.val_dataset = Remote14(root_dir=self.image_dir, is_val=True)
        self.test_dataset = Remote14(root_dir=self.image_dir, is_test=True)

        self.train_loader = DataLoader(
                                self.train_dataset,
                                batch_size=self.bs,
                                shuffle=True,
                                pin_memory=True
                            )
        self.val_loader = DataLoader(
                                self.val_dataset,
                                batch_size=bs,
                                shuffle=False,
                                pin_memory=True
                            )
        self.test_loader = DataLoader(
                                self.test_dataset,
                                batch_size=bs,
                                shuffle=False,
                                pin_memory=True
                            )

        self.classes = self.train_dataset.classes
        self.questions = self._prepare_prompt()
        self.class_anchors = self._prepare_anchors(2)
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
    hparams = {
        'model_name': 'openai/clip-vit-base-patch16', # 'openai/clip-vit-large-patch14-336'
        'adapter_type': 'transformer', # 'mlp', 'transformer', 'mamba'
        'save_path': 'ckpts/adapter.pth',
        'load_path': 'ckpts/adapter_transformer_32_1e-3.pth',
        'device': torch.device("cuda"),
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'bs': 32,
        'image_dir': 'data/remote14',
    }

    classifier = CLIPClassifier(**hparams)

    train_adapter(model_class=classifier, epochs=300)
    # test_adapter(model_class=classifier, split='val')

    # classify_zeroshot(model_class=classifier, split='train' )
    # classify_fewshotshot(model_class=classifier, split='train')
