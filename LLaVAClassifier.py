from typing import List, Dict
import requests
from io import BytesIO
import os
import textwrap
import matplotlib.pyplot as plt
from PIL import Image
from transformers.generation.streamers import TextIteratorStreamer
from transformers import BitsAndBytesConfig, pipeline
from torchvision.transforms import ToPILImage
import torch
from threading import Thread
from sklearn.metrics import classification_report

from torch.utils.data import DataLoader
from datasets import Remote14_raw
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

        self.model_path = "llava-hf/llava-1.5-7b-hf"
        # self.model_path = "liuhaotian/llava-v1.6-mistral-7b"

        # self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        #     model_path = self.model_path, 
        #     model_name=get_model_name_from_path(self.model_path), 
        #     model_base=None, 
        #     load_8bit=False, 
        #     load_4bit=True)
        
        # self.conv_mode = "llava_v0"
        self.writer = SummaryWriter()
        self.image_dir = 'data/remote14'
        self.topil = ToPILImage()
        
        self.train_dataset = Remote14_raw(root_dir=self.image_dir, is_train=True)
        self.val_dataset = Remote14_raw(root_dir=self.image_dir, is_val=True)
        self.test_dataset = Remote14_raw(root_dir=self.image_dir, is_test=True)

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

        self.classes = ['back', 'bottom', 'bottomleftback', 'bottomleftfront', 'bottomrightback', 'bottomrightfront', 'front', 'left', 'right', 
                        'top', 'topleftback', 'topleftfront', 'toprightback', 'toprightfront']
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
    
    def load_image(self, path: str) -> Image:
        if path.startswith('http') or path.startswith('https'):
            response = requests.get(path)
            return Image.open(BytesIO(response.content)).convert('RGB')
        return Image.open(path).convert('RGB')
    
    def generate_text(self, input_ids: torch.Tensor, images_tensor: torch.Tensor, streamer: TextIteratorStreamer) -> str:
        with torch.inference_mode() and torch.cuda.amp.autocast():
            self.model.generate(
                inputs=input_ids,
                images=images_tensor,
                do_sample=True,
                temperature=0.05,
                top_p=1.0,
                max_new_tokens=100,
                streamer=streamer,
            )
    
    def ask_question(self, image: Image, question: str) -> str:
        conv = conv_templates[self.conv_mode].copy()
        inp = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        print(inp)
        print(conv.get_prompt())
        image_tensor = self.image_processor.preprocess(inp, return_tensors='pt', do_rescale=False)['pixel_values'].to(self.model.device)
        input_ids = tokenizer_image_token(question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0)
        thread = Thread(target=self.generate_text, args=(input_ids, image_tensor, streamer))
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
        print(generated_text)
        plt.imshow(image)
        wrapped_title = "\n".join(textwrap.wrap(generated_text, width=100))
        plt.title(wrapped_title)
        plt.show()
        return generated_text
    
    def prompt(self, image: Image, question: str) -> str:
        # image_url = "https://llava-vl.github.io/static/images/view.jpg"
        # image = Image.open(requests.get(image_url, stream=True).raw)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        pipe = pipeline("image-to-text", model=self.model_path, model_kwargs={"quantization_config": quantization_config})
        prompt = "USER: <image>\n" + question + "\nASSISTANT:"

        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
        return outputs[0]["generated_text"]

    def train_adapter(self, epochs: int = 10) -> None:
        # self.model.eval()  
        self.adapter.train()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        pipe = pipeline("image-to-text", model=self.model_path, model_kwargs={"quantization_config": quantization_config}) 
        question = (
            "Tell me from which direction is the photo taken, be as precise as possible and choose from the following options: "
            "0: back, 1: bottom, 2: bottom left back, 3: bottom left front, 4: bottom right back, 5: bottom right front, 6: front, 7: left, 8: right, 9: top, 10: top left back, 11: top left front, 12: top right back, 13: top right front? "
            "Answer solely with the option, not with a sentence."
        )

        # Tell me from which direction is the photo taken, be as precise as possible and choose from the following options: 0: back, 1: bottom, 2: bottom left back, 3: bottom left front, 4: bottom right back, 5: bottom right front, 6: front, 7: left, 8: right, 9: top, 10: top left back, 11: top left front, 12: top right back, 13: top right front? Answer solely with the option, not with a sentence

        for epoch in range(epochs):
            # Training loop
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = batch
                batch_predictions = []
                batch_labels = []
                batch_generated_text = []

                images_pil = [self.topil(image) for image in images]
                for i, image_pil in enumerate(images_pil):
                    prompt = f"USER: <image>\n{question}\nASSISTANT:"
                    generated_text = pipe(image_pil, prompt=prompt, generate_kwargs={"max_new_tokens": 200})[0]["generated_text"]
                    wrapped_text = "\n".join(textwrap.wrap(generated_text, width=40))
                    batch_generated_text.append(wrapped_text)

                    parsed_label = generated_text.split("ASSISTANT:")[-1].strip()
                    wrapped_label = "\n".join(textwrap.wrap(parsed_label, width=40))
                    batch_predictions.append(wrapped_label)

                    gt_label = self.classes[labels[i].cpu().item()]
                    batch_labels.append(gt_label)

                fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
                for j, ax in enumerate(axes):
                    img = images[j].cpu().numpy().transpose((1, 2, 0))
                    ax.imshow(img)
                    ax.set_title(f"GT: {labels[j]}: {batch_labels[j]}, Pred: {batch_predictions[j]}")
                    ax.text(0.5, -0.1, batch_generated_text[j], transform=ax.transAxes, ha='center', va='top', fontsize=12, color="black")
                    ax.axis('off')
                plt.show()

if __name__ == '__main__':
    llava_classifier = LLaVAClassifier(bs=4)
    llava_classifier.train_adapter(epochs=50)
    # llava_classifier.ask_question(llava_classifier.load_image('/home/cc/Downloads/cat-container.jpg'), "What do you see in this photo? Is it a cat? \n")
