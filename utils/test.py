import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from utils.datasets import Remote60
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from torchvision import transforms
from transformers import BitsAndBytesConfig, pipeline
#import transformers
#print(transformers.__file__)
import textwrap
from torchvision.transforms import functional as F
from utils.positions import classes
from utils.misc import hamming_dist, interpolate_color

from utils.misc import parse_output,calculate_mean_std
from PIL import Image 
import numpy as np    

def test_adapter(model_class, split, train_descriptions, val_descriptions, test_descriptions, lam, plot, zeroshot,load_path, load_path_descriptions) -> None:
    fusion = model_class.fusion

    if model_class.embedder == 'clip':
        model_class.clip_model.eval()  
        with torch.no_grad():
            text_inputs = model_class.clip_processor(text=model_class.questions, return_tensors="pt", padding=True).to(model_class.device)
            text_features = model_class.clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(model_class.device)
    elif model_class.embedder == 'blip':
        model_class.blip_model.eval()

    if not zeroshot:
        model_class.adapter_image.eval() 
        model_class.adapter_image.load_state_dict(torch.load(load_path))
        if fusion:
            model_class.adapter_descriptions.eval() 
            model_class.adapter_descriptions.load_state_dict(torch.load(load_path_descriptions))

    if split == 'train':
        train_dataset = Remote60(root_dir=model_class.image_dir, is_train=True, descriptions_file=train_descriptions)
        dataloader = DataLoader(train_dataset, batch_size=model_class.bs, pin_memory=True, num_workers=2) 

    elif split == 'val':
        val_dataset = Remote60(root_dir=model_class.image_dir, is_val=True, descriptions_file=val_descriptions)
        dataloader = DataLoader(val_dataset, batch_size=model_class.bs, pin_memory=True, num_workers=2)
        
    elif split == 'test':
        test_dataset = Remote60(root_dir=model_class.image_dir, is_test=True, descriptions_file=test_descriptions)
        dataloader = DataLoader(test_dataset, batch_size=model_class.bs, pin_memory=True, num_workers=2)

    with torch.no_grad():
        for batch in dataloader:
            if split == 'test':
                images, descriptions = batch
            else:
                images, labels, descriptions = batch

            if model_class.embedder == 'clip':
                with torch.no_grad():
                    if fusion:
                        descriptions_inputs = model_class.clip_processor(text=descriptions, return_tensors="pt", padding=True, truncation=True, max_length=77).to(model_class.device)
                        descriptions_embeds = model_class.clip_model.get_text_features(**descriptions_inputs)
                        descriptions_features = descriptions_embeds / descriptions_embeds.norm(dim=-1, keepdim=True)
                    
                    inputs = model_class.clip_processor(images=images, return_tensors="pt", do_rescale=False).to(model_class.device)
                    image_features = model_class.clip_model.get_image_features(**inputs)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True).to(model_class.device)

                if not zeroshot:
                    image_features = model_class.adapter_image(image_features)
                    if fusion:
                        descriptions_features = model_class.adapter_descriptions(descriptions_features)

            elif model_class.embedder == 'blip':
                with torch.no_grad():
                    images = images.to(model_class.device)

                    text_features = model_class.blip_model(images, model_class.questions, mode='text')[:,0]
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(model_class.device)

                    image_features = model_class.blip_model(images, descriptions, mode='image')
                    image_features = image_features.mean(dim=1)
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True).to(model_class.device)
                    if fusion:
                        descriptions_features = model_class.blip_model(images, descriptions, mode='text')
                        descriptions_features = descriptions_features.mean(dim=1)
                        descriptions_features = descriptions_features / descriptions_features.norm(p=2, dim=-1, keepdim=True).to(model_class.device)

                if not zeroshot:
                    image_features = model_class.adapter_image(image_features)
                    if fusion:
                        descriptions_features = model_class.adapter_descriptions(descriptions_features)

            ###################### Cosine Similarity ##############################################################
            cos_sim = torch.nn.functional.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
            cos_sim = cos_sim / cos_sim.norm(dim=-1, keepdim=True)

            if fusion:
                cos_sim_d = lam * torch.nn.functional.cosine_similarity(descriptions_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
                cos_sim_d = cos_sim_d / cos_sim_d.norm(dim=-1, keepdim=True)

                cos_sim += cos_sim_d
                cos_sim = cos_sim / cos_sim.norm(dim=-1, keepdim=True)

            predicted_classes = torch.argmax(cos_sim, dim=-1)        
            ##############################################################################################################

            model_class.metrics[split]['preds'].extend(predicted_classes.tolist())
            if split != 'test':
                model_class.metrics[split]['gts'].extend(labels.cpu().tolist())

            if plot:
                num_images = len(images) 
                num_rows = 2
                num_cols = 5

                fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(30, 30))  
                for ax_row in axes:
                    for ax in ax_row:
                        ax.axis('off')

                for i in range(num_images):
                    img = images[i].cpu().numpy().transpose((1, 2, 0))
                    pred = predicted_classes[i]
                    pred_class = classes[pred]  

                    if split != 'test':
                        gt = labels[i]
                        gt_class = classes[gt]  

                    row = i // num_cols  
                    col = i % num_cols  
                    ax = axes[row, col]  
                    ax.imshow(img)

                    if split == 'test':
                        ax.text(0.5, -0.02, pred_class, transform=ax.transAxes, ha='center', va='top', fontsize=15, color="blue")
                    else:
                        hamming_distance = hamming_dist(pred, gt).item()
                        color = interpolate_color(hamming_distance)
                        ax.text(0.5, -0.02, pred_class, transform=ax.transAxes, ha='center', va='top', fontsize=15, color=color)
                        ax.set_title(gt_class, fontsize=15, color="black", pad=1)
                        wrapped_text = "\n".join(textwrap.wrap(f"d = {hamming_dist(pred, gt)}", width=25))
                        ax.text(0.5, -0.2, wrapped_text, transform=ax.transAxes, ha='center', va='top', fontsize=15, color=color)

                plt.show()
    
    if split != 'test':
        acc = sum([1 for gt, pred in zip(model_class.metrics[split]['gts'], model_class.metrics[split]['preds']) if gt == pred]) / len(model_class.metrics[split]['gts'])
        print(f"Accuracy: {acc}")
        print(model_class.metrics)

        print(classification_report(model_class.metrics[split]['gts'], model_class.metrics[split]['preds'], target_names=classes))
        model_class._reset_metrics()


def inference_single_image(model_class, image_path, lam, plot, zeroshot,load_path, load_path_descriptions, llava_path) -> None:
    desired_size = (800,800)
    transform = transforms.Compose([
            # transforms.CenterCrop((int(self.desired_size[0] * 1), int(self.desired_size[1] * 1))),
            transforms.Resize(desired_size),
            transforms.ToTensor()
        ])
    fusion = model_class.fusion

    if model_class.embedder == 'clip':
        model_class.clip_model.eval()  
        with torch.no_grad():
            text_inputs = model_class.clip_processor(text=model_class.questions, return_tensors="pt", padding=True).to(model_class.device)
            text_features = model_class.clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(model_class.device)
    elif model_class.embedder == 'blip':
        model_class.blip_model.eval()

    if not zeroshot:
        model_class.adapter_image.eval() 
        model_class.adapter_image.load_state_dict(torch.load(load_path))
        if fusion:
            model_class.adapter_descriptions.eval() 
            model_class.adapter_descriptions.load_state_dict(torch.load(load_path_descriptions))

    question = "Describe the romote in the image, not the background and hand. For example, you can describe the orientation."
    # question = "describe the image in the image."
    prompt = "USER: <image>\n" + question + "\nASSISTANT:"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    pipe = pipeline("image-to-text", model=llava_path, model_kwargs={"quantization_config": quantization_config}) 

    # m = pipe.model
    # for name, param in m.named_parameters():
    #     print(f"Pipeline Model Parameter: {name}, Dtype: {param.dtype}")

    with torch.no_grad():
        img = Image.open(image_path)

        description = pipe(img, prompt=prompt, generate_kwargs={"max_new_tokens": 77})
        # print(pipe.model)
        description = parse_output(description[0]["generated_text"]) 
        image = transform(img.convert("RGB")).unsqueeze(0).to(model_class.device)

    if model_class.embedder == 'clip':
        with torch.no_grad():
            if fusion:
                descriptions_inputs = model_class.clip_processor(text=description, return_tensors="pt", padding=True, truncation=True, max_length=77).to(model_class.device)
                descriptions_embeds = model_class.clip_model.get_text_features(**descriptions_inputs)
                descriptions_features = descriptions_embeds / descriptions_embeds.norm(dim=-1, keepdim=True)
            
            inputs = model_class.clip_processor(images=image, return_tensors="pt", do_rescale=False).to(model_class.device)
            image_features = model_class.clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).to(model_class.device)

        if not zeroshot:
            image_features = model_class.adapter_image(image_features)
            if fusion:
                descriptions_features = model_class.adapter_descriptions(descriptions_features)

    elif model_class.embedder == 'blip':
        with torch.no_grad():
            text_features = model_class.blip_model(image, model_class.questions, mode='text')[:,0]
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(model_class.device)

            image_features = model_class.blip_model(image, description, mode='image')
            image_features = image_features.mean(dim=1)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True).to(model_class.device)
            if fusion:
                descriptions_features = model_class.blip_model(image, description, mode='text')
                descriptions_features = descriptions_features.mean(dim=1)
                descriptions_features = descriptions_features / descriptions_features.norm(p=2, dim=-1, keepdim=True).to(model_class.device)

        if not zeroshot:
            image_features = model_class.adapter_image(image_features)
            if fusion:
                descriptions_features = model_class.adapter_descriptions(descriptions_features)

    ###################### Cosine Similarity ##############################################################
    cos_sim = torch.nn.functional.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
    cos_sim = cos_sim / cos_sim.norm(dim=-1, keepdim=True)

    if fusion:
        cos_sim_d = lam * torch.nn.functional.cosine_similarity(descriptions_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
        cos_sim_d = cos_sim_d / cos_sim_d.norm(dim=-1, keepdim=True)

        cos_sim += cos_sim_d
        cos_sim = cos_sim / cos_sim.norm(dim=-1, keepdim=True)

    predicted_classes = torch.argmax(cos_sim, dim=-1)        
    ##############################################################################################################

    if plot:
        img_width, img_height = img.size
        max_char_width = 100  
        wrapped_description = "\n".join(textwrap.wrap(description, max_char_width))
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        pred_label = classes[predicted_classes.item()]
        plt.title(pred_label, fontsize=20, color="blue", pad=10, fontweight='bold')
        plt.text(img_width // 2, img_height + 10, wrapped_description, ha='center', va='top', fontsize=12, color="black", fontstyle='italic')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

