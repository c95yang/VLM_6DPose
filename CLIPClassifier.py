from typing import List, Dict
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from datasets import Remote14
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import random
import torch.nn as nn
import torch.optim as optim
from adapter import MLPAdapter, TransformerAdapter, MambaAdapter
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

class CLIPClassifier:
    def __init__(self, device: torch.device, bs: int, model_name: str, adapter_type: str, load_path: str, save_path: str) -> None:
        self.save_path = save_path
        self.load_path = load_path
        self.adapter_type = adapter_type
        self.device = device
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(self.model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
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
        self.class_anchors = self._prepare_anchors(2)
        self.metrics = self._reset_metrics()

        if self.adapter_type == 'mlp':
            self.adapter = MLPAdapter().to(device)
        elif self.adapter_type == 'transformer':
            self.adapter = TransformerAdapter().to(device)

        self.optimizer = optim.Adam(self.adapter.parameters(), lr=1e-5, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()

        self.warmup_epochs=3
        self.scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=3, eta_min=1e-6)
        self.scheduler = GradualWarmupScheduler(self.optimizer, multiplier=1, total_epoch=self.warmup_epochs, after_scheduler=self.scheduler_cosine)

    def _reset_metrics(self) -> Dict[str, Dict[str, List[int]]]:
        self.metrics = {'train': {'gts': [], 'preds': []}, 'val': {'gts': [], 'preds': []}, 'test': {'gts': [], 'preds': []}}
        return self.metrics
    
    def _prepare_multiple_prompt(self) -> List[str]:
        questions = [f"A remote control device observed from {CLASS_NAME} direction." for CLASS_NAME in self.classes]
        #questions.extend([f"A remote observed from {CLASS_NAME} direction." for CLASS_NAME in self.classes])
        #questions.extend([f"A photo of remote taken from {CLASS_NAME} direction." for CLASS_NAME in self.classes])
        return questions

    def classify_zeroshot(self, split: str = 'train') -> None:
        dataloader = self.train_loader if split == 'train' else self.test_loader

        batch_counter = 0
        for batch in dataloader:
            batch_counter += 1
            images, label = batch

            texts = self._prepare_multiple_prompt()

            inputs = self.processor(texts, images, return_tensors="pt", padding=True)
            inputs.to(self.model.device)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image

            probs = logits_per_image.softmax(dim=1)
            preds = probs.argmax(dim=-1)
            preds = preds % len(self.classes)

            gt_classes = torch.tensor(label).to(self.model.device)

            self.metrics[split]['gts'].extend(gt_classes.cpu().tolist())
            self.metrics[split]['preds'].extend(preds.cpu().tolist())
            print(self.metrics)

        print(classification_report(self.metrics[split]['gts'], self.metrics[split]['preds'], target_names=self.classes))
        self._reset_metrics()

    def _prepare_anchors(self, few_shot_n: int) -> Dict[str, List[Image.Image]]:
        class_anchors: Dict[str, List[Image.Image]] = {c: [] for c in self.classes}  # Fill this dictionary with the anchor images for each class

        for class_name in self.classes:
            # Get the indices of the images of the class
            class_indices = [idx for idx, label in enumerate(self.train_dataset.labels) if label == class_name]

            selected_indices = random.sample(class_indices, few_shot_n)
            for idx in selected_indices:
                image = self.train_dataset[idx][0] 
                class_anchors[class_name].append(image)

        return class_anchors

    def classify_fewshotshot(self, split: str = 'train') -> None:
        dataloader = self.train_loader if split == 'train' else self.test_loader
        anchor_embeddings = []  # At the end, should have the shape (self.classes, embed_dim)

        for class_name, anchor_images in self.class_anchors.items():
            with torch.no_grad():
                inputs = self.processor(images=anchor_images, return_tensors="pt")
                inputs.to(self.model.device)
                embeds = self.model.get_image_features(**inputs)
                embeds = embeds.mean(dim=0)
                embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                anchor_embeddings.append(embeds)

        anchor_embeddings = torch.stack(anchor_embeddings)

        batch_counter = 0

        for batch in dataloader:
            batch_counter += 1
            images, labels = batch

            inputs = self.processor(images=images, return_tensors="pt")
            inputs.to(self.model.device)
            embeds = self.model.get_image_features(**inputs)
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)

            # cosine similarity calculation
            similarities = torch.matmul(embeds, anchor_embeddings.transpose(0, 1))
            norm_image_embeds = torch.norm(embeds, p=2, dim=-1, keepdim=True)
            norm_anchor_embeddings = torch.norm(anchor_embeddings, p=2, dim=-1, keepdim=True)
            cos_sim = torch.div(similarities, torch.matmul(norm_image_embeds, norm_anchor_embeddings.transpose(0, 1)))
            preds = torch.argmax(cos_sim, dim=-1)
            probs = torch.softmax(cos_sim, dim=-1)
            #print("probs: ", probs)
            
            self.metrics[split]['preds'].extend(preds.tolist())
            self.metrics[split]['gts'].extend(labels.cpu().tolist())
            print(self.metrics)
             
        print(classification_report(self.metrics[split]['gts'], self.metrics[split]['preds'], target_names=self.classes))
        self._reset_metrics()

    def classify_withadapter(self, split: str = 'val') -> None:
        if split == 'val':
            dataloader = self.val_loader
        elif split == 'test':
            dataloader = self.test_loader
        else:
            dataloader = self.train_loader

        self.adapter.eval()
        self.adapter.load_state_dict(torch.load(self.load_path))

        for batch in dataloader:
            images, labels = batch

            inputs = self.processor(images=images, return_tensors="pt", do_rescale=False)
            inputs.to(self.model.device)
            print("inputs: ", inputs)
            embeds = self.model.get_image_features(**inputs)
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            image_features = self.adapter(embeds) #([266, 512])

            text_inputs = self.processor(text=self.questions, return_tensors="pt", padding=True).to(device)
            text_features = self.model.get_text_features(**text_inputs) #([14, 512])

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarities = torch.matmul(image_features, text_features.transpose(0, 1)) #[266,14]
            norm_text_features = torch.norm(text_features, p=2, dim=-1, keepdim=True)
            norm_image_features = torch.norm(image_features, p=2, dim=-1, keepdim=True)
            cos_sim = torch.div(similarities, torch.matmul(norm_image_features, norm_text_features.transpose(0, 1))) #[266,14]

            predicted_classes = torch.argmax(cos_sim, dim=-1)

            self.metrics[split]['preds'].extend(predicted_classes.tolist())
            self.metrics[split]['gts'].extend(labels.cpu().tolist())


            acc = sum([1 for gt, pred in zip(self.metrics[split]['gts'], self.metrics[split]['preds']) if gt == pred]) / len(self.metrics[split]['gts'])
            print(f"Accuracy: {acc}")
            print(self.metrics)
            
            num_images = len(images) 
            num_rows = 2
            num_cols = 8

            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 20))  
            for ax_row in axes:
                for ax in ax_row:
                    ax.axis('off')

            for i in range(num_images):
                img = images[i].cpu().numpy().transpose((1, 2, 0))

                pred_label = self.classes[predicted_classes[i]]
                gt_label = self.classes[labels[i].cpu().item()]  

                row = i // num_cols  
                col = i % num_cols  
                ax = axes[row, col]  
                ax.imshow(img)
                ax.set_title(pred_label, fontsize=12, color="green" if pred_label == gt_label else "red", pad=10)
                ax.text(0.5, -0.1, gt_label, transform=ax.transAxes, ha='center', va='top', fontsize=12, color="black")

            plt.show()

        print(classification_report(self.metrics[split]['gts'], self.metrics[split]['preds'], target_names=self.classes))
        self._reset_metrics()

    def train_adapter(self, epochs: int = 10) -> None:
        self.model.eval()  # Freeze the CLIP model
        self.adapter.train()

        best_val_loss = float('inf')  

        for epoch in range(epochs):
            # Training loop
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = batch
                #print("images: ", images)

                inputs = self.processor(images=images, return_tensors="pt", do_rescale=False).to(self.device)
                #inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                #print("inputs: ", inputs)
                embeds = self.model.get_image_features(**inputs)
                embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)

                image_features = self.adapter(embeds)

                text_inputs = self.processor(text=self.questions, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    text_features = self.model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                cos_sim = torch.matmul(image_features, text_features.transpose(0, 1))
                predicted_classes = torch.argmax(cos_sim, dim=-1)
                # print(cos_sim)     
                # print("labels: ", labels) 
                # print("predicted_classes:", predicted_classes)

                loss = self.criterion(cos_sim, labels.to(self.device))

                train_loss = loss.item()
                self.optimizer.zero_grad()
                loss.backward()

                # for name, param in self.adapter.named_parameters():
                #     print(name, param.grad)

                self.optimizer.step()
                self.scheduler.step()
                print(f"Lr: {self.optimizer.param_groups[0]['lr']}")

                self.metrics['train']['preds'].extend(predicted_classes.cpu().tolist())
                self.metrics['train']['gts'].extend(labels.cpu().tolist())

            # Validation loop
            self.adapter.eval()  
            with torch.no_grad():
                val_losses = []
                for batch in tqdm(self.val_loader, desc=f"Validation {epoch+1}/{epochs}"):
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)

                    inputs = self.processor(images=images, return_tensors="pt", do_rescale=False).to(self.device)
                    # inputs = self.processor(images=images, return_tensors="pt").to(self.device)
                    embeds = self.model.get_image_features(**inputs)
                    embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                    image_features = self.adapter(embeds)

                    text_inputs = self.processor(text=self.questions, return_tensors="pt", padding=True).to(self.device)
                    with torch.no_grad():
                        text_features = self.model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                    cos_sim = torch.matmul(image_features, text_features.transpose(0, 1))
                    # print("cos_sim: ", cos_sim)

                    predicted_classes = torch.argmax(cos_sim, dim=-1)
                    self.metrics['val']['preds'].extend(predicted_classes.cpu().tolist())
                    self.metrics['val']['gts'].extend(labels.cpu().tolist())

                    val_loss = self.criterion(cos_sim, labels.to(self.device))
                    val_losses.append(val_loss.item())

            train_acc = sum([1 for gt, pred in zip(self.metrics['train']['gts'], self.metrics['train']['preds']) if gt == pred]) / len(self.metrics['train']['gts'])
            val_acc = sum([1 for gt, pred in zip(self.metrics['val']['gts'], self.metrics['val']['preds']) if gt == pred]) / len(self.metrics['val']['gts'])
            self.writer.add_scalars('Acc', {'Train': train_acc, 'Validation': val_acc}, epoch)

            mean_val_loss = sum(val_losses) / len(val_losses)

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save(self.adapter.state_dict(), self.save_path)
                print(f"Model saved at epoch {epoch+1}, with validation loss: {mean_val_loss}, path: {self.save_path}, train_acc: {train_acc}, val_acc: {val_acc}")

            self.writer.add_scalars('Losses', {'Train': train_loss, 'Val': mean_val_loss}, epoch)
            self.metrics = self._reset_metrics()

if __name__ == '__main__':
    model_name = 'openai/clip-vit-base-patch16' # 'openai/clip-vit-large-patch14-336' 
    adapter_type = 'transformer' # 'transformer', 'mamba'
    save_path = 'ckpts/adapter.pth'
    load_path = 'ckpts/adapter_transformer_16_1e-5.pth'
    device = torch.device("cuda") 

    classifier = CLIPClassifier(device, model_name=model_name, bs=16, adapter_type=adapter_type, load_path=load_path, save_path=save_path)

    # classifier.train_adapter(epochs=300)
    classifier.classify_withadapter(split='val')


    # classifier.classify_zeroshot(split='train' )
    # classifier.classify_fewshotshot(split='train')
