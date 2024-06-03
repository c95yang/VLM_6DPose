from typing import List, Dict
import textwrap
import os
from PIL import Image
from transformers.generation.streamers import TextIteratorStreamer
import torch
from threading import Thread
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from datasets import Remote14
from tqdm import tqdm
#from transformers import CLIPProcessor, CLIPModel
import random
import torch.nn as nn
import torch.optim as optim
from utils.clip_adapter import clip
from transformers import CLIPProcessor, CLIPModel
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    get_model_name_from_path,
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token
)
from llava.eval.run_llava import eval_model
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle

from adapter import Adapter
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from warmup_scheduler import GradualWarmupScheduler

class LLaVAClassifier:
    def __init__(self, bs: int) -> None:
        self.save_path = 'adapter.pth'
        self.load_path = 'adapter.pth'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            "liuhaotian/llava-v1.5-7b", 
            model_name="llava-v1.5-7b", 
            model_base=None, 
            load_8bit=False, 
            load_4bit=True)
        
        self.writer = SummaryWriter()

        self.image_dir = 'data/remote14'
        self.train_dataset = Remote14(root_dir=self.image_dir, is_train=True)
        self.val_dataset = Remote14(root_dir=self.image_dir, is_val=True)
        self.test_dataset = Remote14(root_dir=self.image_dir, is_test=True)

        self.train_loader = DataLoader(
                                self.train_dataset,
                                batch_size=bs,
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
        self.questions = self._prepare_multiple_prompt()
        self.metrics = self._reset_metrics()

        self.adapter = Adapter(in_features=338688).to(self.device)
        self.optimizer = optim.Adam(self.adapter.parameters(), lr=1e-1, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        self.warmup_epochs=3
        self.scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=3, eta_min=1e-6)
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=self.warmup_epochs, after_scheduler=self.scheduler_cosine)

    def _reset_metrics(self) -> Dict[str, Dict[str, List[int]]]:
        self.metrics = {'train': {'gts': [], 'preds': []}, 'val': {'gts': [], 'preds': []}, 'test': {'gts': [], 'preds': []}}
        return self.metrics
    
    def _prepare_multiple_prompt(self) -> List[str]:
        questions = [f"A remote control device observed from {CLASS_NAME} direction." for CLASS_NAME in self.classes]
        return questions
    

    def train_adapter(self, epochs: int = 10) -> None:
        self.model.eval()  # Freeze the CLIP model
        self.adapter.train()

        best_val_loss = float('inf')  

        for epoch in range(epochs):
            # Training loop
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = batch

                conv_mode = "llava_v0"
                conv = conv_templates[conv_mode].copy()
                images_tensor = self.image_processor(images, return_tensors='pt')['pixel_values'].to(self.model.device)
                print(images_tensor.shape)
                
                #embeds = images_tensor.flatten(2, 3).flatten(1, 2)
                #image_features = self.adapter(embeds)

                # prompt = "Which direction is the object in the photo observed from? Choose only one direction from the following options: back? \
                #     bottom? bottomleftback? bottomleftfront?bottomrightback? bottomrightfront?\
                #           front? left?right? top? topleftback? topleftfront? toprightback? toprightfront?"
                
                prompt = "Which direction is the object in the photo observed from? "

                inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
                conv.append_message(conv.roles[0], inp)

                conv.append_message(conv.roles[1], None)
                #prompt = conv.get_prompt()
            
                input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)
    
                def generate_text():
                    with torch.inference_mode() and torch.cuda.amp.autocast():
                        self.model.generate(
                            inputs=input_ids,
                            images=images_tensor,
                            do_sample=True,
                            temperature=0.1,
                            top_p=1.0,
                            max_new_tokens=100,
                            streamer=streamer,
                            use_cache=True
                        )
                
                thread = Thread(target=generate_text)
                thread.start()
                prepend_space = False
                generated_text = ""
                
                for new_text in streamer:
                    if new_text == " ":
                        prepend_space = True
                        continue
                    if new_text.endswith(stop_str):
                        new_text = new_text[:-len(stop_str)].strip()
                        prepend_space = False
                    elif prepend_space:
                        new_text = " " + new_text
                        prepend_space = False
                    if len(new_text):
                        generated_text += new_text
                if prepend_space:
                    generated_text += " "
                thread.join()

                print(generated_text)

                text_inputs = self.processor(text=generated_text, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    text_features = self.model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                cos_sim = torch.matmul(image_features, text_features.transpose(0, 1))
                predicted_classes = torch.argmax(cos_sim, dim=-1)

                loss = self.criterion(cos_sim, labels.to(self.device))

                train_loss = loss.item()
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                self.metrics['train']['preds'].extend(predicted_classes.cpu().tolist())
                self.metrics['train']['gts'].extend(labels.cpu().tolist())

            # Validation loop
            self.adapter.eval()  
            with torch.no_grad():
                val_losses = []
                for batch in tqdm(self.val_loader, desc=f"Validation {epoch+1}/{epochs}"):
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    images_tensor = process_images(
                        images,
                        self.image_processor,
                        self.model.config
                    ).to(self.model.device)
                    
                    embeds = images_tensor.flatten(2, 3).flatten(1, 2)
                    embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                    adapter = Adapter(in_features=embeds.shape[-1]).to(self.device)
                    image_features = adapter(embeds)

                    text_inputs = self.processor(text=self.questions, return_tensors="pt", padding=True).to(self.device)
                    with torch.no_grad():
                        text_features = self.model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    cos_sim = torch.matmul(image_features, text_features.transpose(0, 1))

                    predicted_classes = torch.argmax(cos_sim, dim=-1)
                    self.metrics['val']['preds'].extend(predicted_classes.cpu().tolist())
                    self.metrics['val']['gts'].extend(labels.cpu().tolist())

                    val_loss = self.criterion(cos_sim, labels.to(self.device))
                    val_losses.append(val_loss.item())

            train_report = classification_report(self.metrics['train']['gts'], self.metrics['train']['preds'], target_names=self.classes, output_dict=True)
            test_report = classification_report(self.metrics['val']['gts'], self.metrics['val']['preds'], target_names=self.classes, output_dict=True)
            for metric, value in train_report.items():
                if (metric == 'accuracy'):
                    train_acc = value
            for metric, value in test_report.items():
                if (metric == 'accuracy'):
                    val_acc = value
            
            self.writer.add_scalars('Acc', {'Train': train_acc, 'Validation': val_acc}, epoch)

            mean_val_loss = sum(val_losses) / len(val_losses)

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save(self.adapter.state_dict(), self.save_path)
                print(f"Model saved at epoch {epoch+1}, with validation loss: {mean_val_loss}, path: {self.save_path}, train_acc: {train_acc}, val_acc: {val_acc}")

            self.writer.add_scalars('Losses', {'Train': train_loss, 'Val': mean_val_loss}, epoch)
            self.metrics = self._reset_metrics()

if __name__ == '__main__':
    llava_classifier = LLaVAClassifier(bs=16)
    llava_classifier.train_adapter(epochs=50)
    #llava_classifier.classify_withadapter(split='val')
