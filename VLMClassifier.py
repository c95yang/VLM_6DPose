from typing import List, Dict
import torch
import torch.nn as nn
import torch.optim as optim
import enum
from torch.utils.tensorboard import SummaryWriter

from models.adapter import MLPAdapter, TransformerAdapter
from utils.train import train
from utils.test import test_adapter, inference_single_image
from utils.positions import classes
from utils.misc import HammingLoss
# from utils.classify import classify_zeroshot, classify_fewshotshot

from BLIP.models.blip import blip_feature_extractor
from transformers import CLIPProcessor, CLIPModel,  AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from utils.misc import log_memory_usage

class VLMClassifier:
    def __init__(self, 
                 device: torch.device, 
                 dtype: torch.dtype,
                 bs: int, 
                 adapter_image_type: str, 
                 adapter_descriptions_type: str, 
                 save_path: str, 
                 save_path_descriptions: str,
                 lr: float, 
                 weight_decay: float,
                 image_dir: str,
                 in_features: int,
                 fusion: bool,
                 embedder: str
                 ) -> None:
        
        self.device = device
        self.dtype = dtype
        torch.set_default_dtype(self.dtype)

        self.bs = bs
        self.adapter_image_type = adapter_image_type
        self.adapter_descriptions_type = adapter_descriptions_type
        self.save_path = save_path
        self.save_path_descriptions = save_path_descriptions
        self.lr = lr
        self.weight_decay = weight_decay
        self.image_dir = image_dir
        self.warmup_epochs=3
        self.in_features = in_features
        self.dtype = dtype
        self.fusion = fusion
        self.embedder = embedder

        self.writer = SummaryWriter()

        # quant_config = BitsAndBytesConfig(load_in_4bit=True)

        # self.phi3model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True).to(device)
        # self.phi3tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", trust_remote_code=True)

        # self.phi3pipe = pipeline( 
        #     "text-generation", 
        #     model=self.phi3model, 
        #     tokenizer=self.phi3tokenizer, 
        # ) 

        # messages = [ 
        #     # {"role": "system", "content": "You are a helpful AI assistant."}, 
        #     # {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}, 
        #     # {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."}, 
        #     {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"}, 
        # ] 

        # output = self.phi3pipe(messages, **generation_args) 
        # print(output[0]['generated_text']) 

        if self.embedder == 'clip':
            clip_model_name = 'openai/clip-vit-large-patch14-336' # 'openai/clip-vit-large-patch14-336', 'openai/clip-vit-base-patch16'
            self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model.requires_grad_(False)
        elif self.embedder == 'blip':
            model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth'
            self.blip_model = blip_feature_extractor(pretrained=model_url, image_size=800, vit='base').to(device)
            self.blip_model.requires_grad_(False)

        if self.adapter_image_type == 'mlp':
            self.adapter_image = MLPAdapter(in_features=self.in_features, hidden_features=128, dtype=self.dtype).to(device)
        elif self.adapter_image_type == 'transformer':
            self.adapter_image = TransformerAdapter(in_features=self.in_features, hidden_features=128, dtype=self.dtype).to(device)
        self.optimizer_image = optim.Adam(self.adapter_image.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler_image = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_image, T_max=3, eta_min=1e-6)

        if self.fusion:
            if self.adapter_descriptions_type == 'mlp':
                self.adapter_descriptions = MLPAdapter(in_features=self.in_features, hidden_features=128, dtype=self.dtype).to(device)
            elif self.adapter_descriptions_type == 'transformer':
                self.adapter_descriptions = TransformerAdapter(in_features=self.in_features, hidden_features=128, dtype=self.dtype).to(device)
            self.optimizer_descriptions = optim.Adam(self.adapter_descriptions.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.scheduler_descriptions = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_descriptions, T_max=3, eta_min=1e-6)

        # self.mm_projector = nn.Linear(self.in_features, 128).to(device)

        self.questions = self._prepare_prompt()
        self.metrics = self._reset_metrics()
        
        self.criterion = nn.CrossEntropyLoss()
        

    def _reset_metrics(self) -> Dict[str, Dict[str, List[int]]]:
        self.metrics = {'train': {'gts': [], 'preds': []}, 'val': {'gts': [], 'preds': []}, 'test': {'gts': [], 'preds': []}}
        return self.metrics
    
    def _prepare_prompt(self) -> List[str]:
        questions = [f"a remote observed from {CLASS_NAME}" for CLASS_NAME in classes]
        return questions
    
    def get_all_parameters(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, nn.Module):
                for name, param in attr.named_parameters():
                    print(f"Model: {attr_name}, Parameter: {name}, Dtype: {param.dtype}, Device: {param.device}")


if __name__ == '__main__':

    hparams = {
        'save_path': 'ckpts/adapter_image.pth',
        'save_path_descriptions': 'ckpts/adapter_descriptions.pth', 
        'device': torch.device("cuda"),
        'dtype': torch.float32,
        'image_dir': 'data/remote60',
        'in_features': 768, #512 for clip base, 768 for clip large
        'adapter_image_type': 'transformer', # 'mlp', 'transformer'
        'adapter_descriptions_type': 'transformer', # 'mlp', 'transformer'
        'lr': 1e-5,
        'weight_decay': 1e-4,
        'bs': 10, #16
        'fusion': True,
        'embedder': 'clip', #'blip', 'clip'
    }

    classifier = VLMClassifier(**hparams)

    # hparams = {
    #     'model_class': classifier, 
    #     'epochs': 100,
    #     'train_descriptions': "descriptions/train_descriptions.json",
    #     'val_descriptions': "descriptions/val_descriptions.json",
    #     'lam': 1,
    #     'zeroshot': False,
    # }

    # train(**hparams)

    # hparams = {
    #     'model_class': classifier, 
    #     'split': 'val',
    #     'train_descriptions': "descriptions/train_descriptions.json",
    #     'val_descriptions': "descriptions/val_descriptions.json",
    #     'test_descriptions': "descriptions/test_descriptions.json",
    #     'lam': 1,
    #     'plot': True,
    #     'zeroshot': False,
    #     'load_path': 'ckpts/adapter_image_blip_imagetext.pth',
    #     'load_path_descriptions': 'ckpts/adapter_descriptions_clip_imagetext.pth',
    # }

    # test_adapter(**hparams)

    hparams = {
        'model_class': classifier, 
        'image_path': 'data/remote60/test/WhatsApp Image 2024-08-13 at 15.45.51 (5).jpeg',
        'lam': 1,
        'plot': True,
        'zeroshot': False,
        'load_path': 'ckpts/adapter_image_clip_imagetext.pth',
        'load_path_descriptions': 'ckpts/adapter_descriptions_clip_imagetext.pth',
        'llava_path': "llava-hf/llava-1.5-7b-hf"
    }

    inference_single_image(**hparams)

