from typing import List, Dict
import textwrap
import os
import matplotlib.pyplot as plt
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
        
        self.conv_mode = "llava_v0"
        
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
        self.metrics = self._reset_metrics()

        self.adapter = Adapter().to(self.device)
        self.optimizer = optim.Adam(self.adapter.parameters(), lr=1e-1, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        self.warmup_epochs=3
        self.scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=3, eta_min=1e-6)
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=self.warmup_epochs, after_scheduler=self.scheduler_cosine)

    def _reset_metrics(self) -> Dict[str, Dict[str, List[int]]]:
        self.metrics = {'train': {'gts': [], 'preds': []}, 'val': {'gts': [], 'preds': []}, 'test': {'gts': [], 'preds': []}}
        return self.metrics 

    def train_adapter(self, epochs: int = 10) -> None:
        self.model.eval()  # Freeze the CLIP model
        self.adapter.train()

        best_val_loss = float('inf')  

        for epoch in range(epochs):
            # Training loop
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = batch

                
                conv = conv_templates[self.conv_mode].copy()
                images_tensor = self.image_processor.preprocess(images, return_tensors='pt', do_rescale=False)['pixel_values'].to(self.model.device)
                
                # image_features = self.adapter(embeds)

                # prompt = "Which direction is the object in the photo observed from? Only following options: back, \
                #     bottom, bottomleftback, bottomleftfront, bottomrightback, bottomrightfront,\
                #           front, left, right, top, topleftback, topleftfront, toprightback, toprightfront."

                # prompt = "Tell me from which direction is the photo taken. 1: back, 2: bottom, 3: bottom left back, 4: bottom leftf ront, \
                #     5: bottom right back, 6: bottom right front,7: front, 8: left, 9: right, 10: top, 11: top left back, 12: top left front, \
                #     13: top right back, 14: top right front"
                
                # prompt = "Tell me from which direction is the photo taken. back, bottom, bottom left back, bottom left front, \
                #     bottom right back, bottom right front, front, left, right, top, top left back, top left front, \
                #     top right back, top right front"
                
                # prompt = "Tell me from which direction is the photo taken: back, bottom, bottom left back, bottom left front, \
                #      bottom right back, bottom right front, front, left, right, top, top left back, top left front, \
                #      top right back, top right front?"
                
                #prompt = "Which direction is the photo taken from: back, bottom, front, left, right, top?"
                
                #prompt = "Which direction is the object in the photo observed from?\n"

                #prompt = "Tell me from which direction is the photo taken. Answer shortly in a word. \n"

                # prompt = "A remote is shown in the photo. In one or two words, tell me from which direction is the camera from. Only following options: back, \
                #     bottom, bottomleftback, bottomleftfront, bottomrightback, bottomrightfront,\
                #     front, left, right, top, topleftback, topleftfront, toprightback, toprightfront."
                
                prompt = "Describe what you see in the photo. Tell me in a short phrase the camera position. \n"


                inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                # prompt = conv.get_prompt()
            
                input_ids = tokenizer_image_token(inp, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                #keywords = [stop_str]
                streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)
    
                def generate_text():
                    with torch.inference_mode() and torch.cuda.amp.autocast():
                        self.model.generate(
                            inputs=input_ids,
                            images=images_tensor,
                            do_sample=True,
                            temperature=0.1,
                            top_p=1.0,
                            max_new_tokens=10,
                            streamer=streamer,
                            #use_cache=True
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

                generated_text = generated_text.replace("</s>", "").strip()
                img = images[0].cpu().numpy().transpose((1, 2, 0))
                gt_label = self.classes[labels[0].cpu().item()]  
                plt.imshow(img)
                plt.title(f"GT: {gt_label}, Pred: {generated_text}")
                plt.show()

            #     text_inputs = self.processor(text=generated_text, return_tensors="pt", padding=True).to(self.device)
            #     with torch.no_grad():
            #         text_features = self.model.get_text_features(**text_inputs)
            #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            #     cos_sim = torch.matmul(image_features, text_features.transpose(0, 1))
            #     predicted_classes = torch.argmax(cos_sim, dim=-1)

            #     loss = self.criterion(cos_sim, labels.to(self.device))

            #     train_loss = loss.item()
            #     self.optimizer.zero_grad()
            #     loss.backward()

            #     self.optimizer.step()
            #     self.scheduler.step()

            #     self.metrics['train']['preds'].extend(predicted_classes.cpu().tolist())
            #     self.metrics['train']['gts'].extend(labels.cpu().tolist())

            # # Validation loop
            # self.adapter.eval()  
            # with torch.no_grad():
            #     val_losses = []
            #     for batch in tqdm(self.val_loader, desc=f"Validation {epoch+1}/{epochs}"):
            #         images, labels = batch
            #         images, labels = images.to(self.device), labels.to(self.device)
                    
            #         conv = conv_templates[self.conv_mode].copy()
            #         images_tensor = self.image_processor.preprocess(images, return_tensors='pt', do_rescale=False)['pixel_values'].to(self.model.device)
                    
            #         # image_features = self.adapter(embeds)

            #         # prompt = "Which direction is the object in the photo observed from? Only following options: back, \
            #         #     bottom, bottomleftback, bottomleftfront, bottomrightback, bottomrightfront,\
            #         #           front, left, right, top, topleftback, topleftfront, toprightback, toprightfront."
                    
            #         prompt = "Which direction is the object in the photo observed from?\n"
            #         # prompt = "Describe the picture."

            #         inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
            #         conv.append_message(conv.roles[0], inp)
            #         conv.append_message(conv.roles[1], None)
            #         #prompt = conv.get_prompt()
                
            #         input_ids = tokenizer_image_token(inp, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            #         stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            #         #keywords = [stop_str]
            #         streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)
        
            #         def generate_text():
            #             with torch.inference_mode() and torch.cuda.amp.autocast():
            #                 self.model.generate(
            #                     inputs=input_ids,
            #                     images=images_tensor,
            #                     do_sample=True,
            #                     temperature=0.1,
            #                     top_p=1.0,
            #                     max_new_tokens=100,
            #                     streamer=streamer,
            #                     #use_cache=True
            #                 )
                    
            #         thread = Thread(target=generate_text)
            #         thread.start()
            #         prepend_space = False
            #         generated_text = ""
                    
            #         for new_text in streamer:
            #             if new_text == " ":
            #                 prepend_space = True
            #                 continue
            #             if new_text.endswith(stop_str):
            #                 new_text = new_text[:-len(stop_str)].strip()
            #                 prepend_space = False
            #             elif prepend_space:
            #                 new_text = " " + new_text
            #                 prepend_space = False
            #             if len(new_text):
            #                 generated_text += new_text
            #         if prepend_space:
            #             generated_text += " "
            #         thread.join()

            #         print(generated_text)

            #         text_inputs = self.processor(text=self.questions, return_tensors="pt", padding=True).to(self.device)
            #         with torch.no_grad():
            #             text_features = self.model.get_text_features(**text_inputs)
            #         text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            #         cos_sim = torch.matmul(image_features, text_features.transpose(0, 1))

            #         predicted_classes = torch.argmax(cos_sim, dim=-1)
            #         self.metrics['val']['preds'].extend(predicted_classes.cpu().tolist())
            #         self.metrics['val']['gts'].extend(labels.cpu().tolist())

            #         val_loss = self.criterion(cos_sim, labels.to(self.device))
            #         val_losses.append(val_loss.item())

            # train_acc = sum([1 for gt, pred in zip(self.metrics['train']['gts'], self.metrics['train']['preds']) if gt == pred]) / len(self.metrics['train']['gts'])
            # val_acc = sum([1 for gt, pred in zip(self.metrics['val']['gts'], self.metrics['val']['preds']) if gt == pred]) / len(self.metrics['val']['gts'])
            # self.writer.add_scalars('Acc', {'Train': train_acc, 'Validation': val_acc}, epoch)

            # mean_val_loss = sum(val_losses) / len(val_losses)

            # if mean_val_loss < best_val_loss:
            #     best_val_loss = mean_val_loss
            #     torch.save(self.adapter.state_dict(), self.save_path)
            #     print(f"Model saved at epoch {epoch+1}, with validation loss: {mean_val_loss}, path: {self.save_path}, train_acc: {train_acc}, val_acc: {val_acc}")

            # self.writer.add_scalars('Losses', {'Train': train_loss, 'Val': mean_val_loss}, epoch)
            # self.metrics = self._reset_metrics()

if __name__ == '__main__':
    llava_classifier = LLaVAClassifier(bs=1)
    llava_classifier.train_adapter(epochs=50)
    #llava_classifier.classify_withadapter(split='val')
