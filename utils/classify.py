from typing import List, Dict
from PIL import Image
import torch
from sklearn.metrics import classification_report
import random

def classify_zeroshot(model_class, split: str = 'train') -> None:
    dataloader = model_class.train_loader if split == 'train' else model_class.test_loader

    batch_counter = 0
    for batch in dataloader:
        batch_counter += 1
        images, label = batch

        texts = model_class._prepare_multiple_prompt()

        inputs = model_class.processor(texts, images, return_tensors="pt", padding=True)
        inputs.to(model_class.model.device)
        outputs = model_class.model(**inputs)
        logits_per_image = outputs.logits_per_image

        probs = logits_per_image.softmax(dim=1)
        preds = probs.argmax(dim=-1)
        preds = preds % len(model_class.classes)

        gt_classes = torch.tensor(label).to(model_class.model.device)

        model_class.metrics[split]['gts'].extend(gt_classes.cpu().tolist())
        model_class.metrics[split]['preds'].extend(preds.cpu().tolist())
        print(model_class.metrics)

    print(classification_report(model_class.metrics[split]['gts'], model_class.metrics[split]['preds'], target_names=model_class.classes))
    model_class._reset_metrics()
        
def classify_fewshotshot(model_class, split: str = 'train') -> None:
    dataloader = model_class.train_loader if split == 'train' else model_class.test_loader
    anchor_embeddings = []  # At the end, should have the shape (model_class.classes, embed_dim)
    anchors = _prepare_anchors(model_class, few_shot_n=5)

    for class_name, anchor_images in anchors.items():
        with torch.no_grad():
            inputs = model_class.processor(images=anchor_images, return_tensors="pt")
            inputs.to(model_class.model.device)
            embeds = model_class.model.get_image_features(**inputs)
            embeds = embeds.mean(dim=0)
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            anchor_embeddings.append(embeds)

    anchor_embeddings = torch.stack(anchor_embeddings)

    batch_counter = 0

    for batch in dataloader:
        batch_counter += 1
        images, labels = batch

        inputs = model_class.processor(images=images, return_tensors="pt")
        inputs.to(model_class.model.device)
        embeds = model_class.model.get_image_features(**inputs)
        embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity calculation
        similarities = torch.matmul(embeds, anchor_embeddings.transpose(0, 1))
        norm_image_embeds = torch.norm(embeds, p=2, dim=-1, keepdim=True)
        norm_anchor_embeddings = torch.norm(anchor_embeddings, p=2, dim=-1, keepdim=True)
        cos_sim = torch.div(similarities, torch.matmul(norm_image_embeds, norm_anchor_embeddings.transpose(0, 1)))
        preds = torch.argmax(cos_sim, dim=-1)
        probs = torch.softmax(cos_sim, dim=-1)
        #print("probs: ", probs)
        
        model_class.metrics[split]['preds'].extend(preds.tolist())
        model_class.metrics[split]['gts'].extend(labels.cpu().tolist())
        print(model_class.metrics)
            
    print(classification_report(model_class.metrics[split]['gts'], model_class.metrics[split]['preds'], target_names=model_class.classes))
    model_class._reset_metrics()

def _prepare_anchors(model_class, few_shot_n: int) -> Dict[str, List[Image.Image]]:
    class_anchors: Dict[str, List[Image.Image]] = {c: [] for c in model_class.classes}  # Fill this dictionary with the anchor images for each class

    for class_name in model_class.classes:
        class_indices = [idx for idx, label in enumerate(model_class.train_dataset.labels) if label == class_name]

        selected_indices = random.sample(class_indices, few_shot_n)
        for idx in selected_indices:
            image = model_class.train_dataset[idx][0] 
            class_anchors[class_name].append(image)

    return class_anchors