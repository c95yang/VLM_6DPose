import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from utils.datasets import Remote14
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from transformers import BitsAndBytesConfig, pipeline
#import transformers
#print(transformers.__file__)
import textwrap
from torchvision.transforms import functional as F

from utils.misc import parse_output
from PIL import Image 
import numpy as np    

def test_adapter(model_class, split, plot) -> None:
    model_class.clip_model.eval()

    model_class.adapter_image.eval()
    model_class.adapter_image.load_state_dict(torch.load(model_class.load_path))
    model_class.adapter_descriptions.eval()
    model_class.adapter_descriptions.load_state_dict(torch.load(model_class.load_path_descriptions))

    if split == 'train':
        train_dataset = Remote14(root_dir=model_class.image_dir, is_train=True, descriptions_file="train_descriptions_concise.json")
        dataloader = DataLoader(train_dataset, batch_size=model_class.bs, shuffle=True, pin_memory=True)
    elif split == 'val':
        val_dataset = Remote14(root_dir=model_class.image_dir, is_val=True, descriptions_file="val_descriptions_concise.json")
        dataloader = DataLoader(val_dataset, batch_size=model_class.bs, shuffle=False, pin_memory=True)
    elif split == 'test':
        test_dataset = Remote14(root_dir=model_class.image_dir, is_test=True, descriptions_file="test_descriptions_concise.json")
        dataloader = DataLoader(test_dataset, batch_size=model_class.bs, shuffle=False, pin_memory=True)

    with torch.no_grad():
        for batch in dataloader:
            images, labels, descriptions = batch

            ###################### images embeds ########################################################################
            inputs = model_class.processor(images=images, return_tensors="pt", do_rescale=False).to(model_class.device)
            embeds = model_class.clip_model.get_image_features(**inputs)
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            image_features = model_class.adapter_image(embeds)
            ##############################################################################################################

            ########### llava descriptions embeds ########################################################################
            descriptions_inputs = model_class.processor(text=descriptions, return_tensors="pt", padding=True, truncation=True, max_length=77).to(model_class.device)
            descriptions_embeds = model_class.clip_model.get_text_features(**descriptions_inputs)
            descriptions_embeds = descriptions_embeds / descriptions_embeds.norm(dim=-1, keepdim=True)
            descriptions_features = model_class.adapter_descriptions(descriptions_embeds)
            ##############################################################################################################

            ###################### questions embeds ######################################################################
            text_inputs = model_class.processor(text=model_class.questions, return_tensors="pt", padding=True).to(model_class.device)
            text_features = model_class.clip_model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            ##############################################################################################################

            ###################### Cosine Similarity fusion ##############################################################
            cos_sim_image = torch.matmul(image_features, text_features.transpose(0, 1))
            #print(f"cos_sim_image: {cos_sim_image}")
            cos_sim_text= torch.matmul(descriptions_features, text_features.transpose(0, 1))
            #print(f"cos_sim_text: {cos_sim_text}")
            cos_sim = cos_sim_image + cos_sim_text
            #print(f"cos_sim: {cos_sim}")
            predicted_classes = torch.argmax(cos_sim, dim=-1)
            ##############################################################################################################

            model_class.metrics[split]['preds'].extend(predicted_classes.tolist())
            model_class.metrics[split]['gts'].extend(labels.cpu().tolist())


            acc = sum([1 for gt, pred in zip(model_class.metrics[split]['gts'], model_class.metrics[split]['preds']) if gt == pred]) / len(model_class.metrics[split]['gts'])
            print(f"Accuracy: {acc}")
            print(model_class.metrics)

            if plot:
                num_images = len(images) 
                num_rows = 2
                num_cols = 8

                fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 20))  
                for ax_row in axes:
                    for ax in ax_row:
                        ax.axis('off')

                for i in range(num_images):
                    img = images[i].cpu().numpy().transpose((1, 2, 0))
                    #img = images[i]

                    pred_label = model_class.classes[predicted_classes[i]]
                    gt_label = model_class.classes[labels[i].cpu().item()]  

                    row = i // num_cols  
                    col = i % num_cols  
                    ax = axes[row, col]  
                    ax.imshow(img)
                    ax.set_title(pred_label, fontsize=10, color="green" if pred_label == gt_label else "red", pad=1)
                    ax.text(0.5, -0.02, gt_label, transform=ax.transAxes, ha='center', va='top', fontsize=10, color="black")
                    wrapped_text = "\n".join(textwrap.wrap(descriptions[i], width=25))
                    ax.text(0.5, -0.2, wrapped_text, transform=ax.transAxes, ha='center', va='top', fontsize=10, color="black")

                plt.show()

    print(classification_report(model_class.metrics[split]['gts'], model_class.metrics[split]['preds'], target_names=model_class.classes))
    model_class._reset_metrics()


# def test_adapter(model_class, split, path, plot) -> None:
#     if split == 'val':
#         val_dataset = Remote14(root_dir=model_class.image_dir, is_val=True)
#         dataloader = DataLoader(val_dataset, batch_size=model_class.bs, shuffle=False, pin_memory=True)
#     elif split == 'test':
#         test_dataset = Remote14(root_dir=model_class.image_dir, is_test=True)
#         dataloader = DataLoader(test_dataset, batch_size=model_class.bs, shuffle=False, pin_memory=True)
#     elif split == 'train':
#         train_dataset = Remote14(root_dir=model_class.image_dir, is_train=True)
#         dataloader = DataLoader(train_dataset, batch_size=model_class.bs, shuffle=True, pin_memory=True)

#     question = "Tell me from which direction is the remote in the image observed, \
#     using options such as front, back, left, right, top, bottom, and their combinations. Provide several options if necessary."

#     prompt = "USER: <image>\n" + question + "\nASSISTANT:"
#     topil = ToPILImage()

#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16
#     )
#     pipe = pipeline("image-to-text", model=path, model_kwargs={"quantization_config": quantization_config}) 

#     model_class.clip_model.eval()

#     model_class.adapter_image.eval()
#     model_class.adapter_image.load_state_dict(torch.load(model_class.load_path))
#     model_class.adapter_descriptions.eval()
#     model_class.adapter_descriptions.load_state_dict(torch.load(model_class.load_path_descriptions))

#     with torch.no_grad():
#         for batch in dataloader:
#             images, labels = batch

#             ###################### images embeds ########################################################################
#             inputs = model_class.processor(images=images, return_tensors="pt", do_rescale=False).to(model_class.device)
#             embeds = model_class.clip_model.get_image_features(**inputs)
#             embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
#             image_features = model_class.adapter_image(embeds)
#             ##############################################################################################################

#             ########### llava descriptions embeds ########################################################################
#             images = [topil(image) for image in images]

#             descriptions = pipe(images, prompt=prompt, generate_kwargs={"max_new_tokens": 77, "min_new_tokens":50})

#             descriptions = [parse_output(description[0]["generated_text"]) for description in descriptions]
#             descriptions_inputs = model_class.processor(text=descriptions, return_tensors="pt", padding=True, truncation=True, max_length=77).to(model_class.device)
#             descriptions_embeds = model_class.clip_model.get_text_features(**descriptions_inputs)
#             descriptions_embeds = descriptions_embeds / descriptions_embeds.norm(dim=-1, keepdim=True)
#             descriptions_features = model_class.adapter_descriptions(descriptions_embeds)
#             ##############################################################################################################

#             ###################### questions embeds ######################################################################
#             text_inputs = model_class.processor(text=model_class.questions, return_tensors="pt", padding=True).to(model_class.device)
#             text_features = model_class.clip_model.get_text_features(**text_inputs)
#             text_features = text_features / text_features.norm(dim=-1, keepdim=True)
#             ##############################################################################################################

#             ###################### Cosine Similarity fusion ##############################################################
#             cos_sim_image = torch.matmul(image_features, text_features.transpose(0, 1))
#             #print(f"cos_sim_image: {cos_sim_image}")
#             cos_sim_text= torch.matmul(descriptions_features, text_features.transpose(0, 1))
#             #print(f"cos_sim_text: {cos_sim_text}")
#             cos_sim = cos_sim_image + cos_sim_text
#             #print(f"cos_sim: {cos_sim}")
#             predicted_classes = torch.argmax(cos_sim, dim=-1)
#             ##############################################################################################################

#             model_class.metrics[split]['preds'].extend(predicted_classes.tolist())
#             model_class.metrics[split]['gts'].extend(labels.cpu().tolist())


#             acc = sum([1 for gt, pred in zip(model_class.metrics[split]['gts'], model_class.metrics[split]['preds']) if gt == pred]) / len(model_class.metrics[split]['gts'])
#             print(f"Accuracy: {acc}")
#             print(model_class.metrics)

#             if plot:
#                 num_images = len(images) 
#                 num_rows = 2
#                 num_cols = 8

#                 fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 20))  
#                 for ax_row in axes:
#                     for ax in ax_row:
#                         ax.axis('off')

#                 for i in range(num_images):
#                     # img = images[i].cpu().numpy().transpose((1, 2, 0))
#                     img = images[i]

#                     pred_label = model_class.classes[predicted_classes[i]]
#                     gt_label = model_class.classes[labels[i].cpu().item()]  

#                     row = i // num_cols  
#                     col = i % num_cols  
#                     ax = axes[row, col]  
#                     ax.imshow(img)
#                     ax.set_title(pred_label, fontsize=10, color="green" if pred_label == gt_label else "red", pad=1)
#                     ax.text(0.5, -0.02, gt_label, transform=ax.transAxes, ha='center', va='top', fontsize=10, color="black")
#                     wrapped_text = "\n".join(textwrap.wrap(descriptions[i], width=25))
#                     ax.text(0.5, -0.2, wrapped_text, transform=ax.transAxes, ha='center', va='top', fontsize=10, color="black")

#                 plt.show()

#     print(classification_report(model_class.metrics[split]['gts'], model_class.metrics[split]['preds'], target_names=model_class.classes))
#     model_class._reset_metrics()


def inference_single_image(model_class, image_path, plot) -> None:
    model_class.clip_model.eval()

    model_class.adapter_image.eval()
    model_class.adapter_image.load_state_dict(torch.load(model_class.load_path))
    model_class.adapter_descriptions.eval()
    model_class.adapter_descriptions.load_state_dict(torch.load(model_class.load_path_descriptions))

    question = "Tell me from which direction is the remote in the image observed, \
    using options such as front, back, left, right, top, bottom, and their combinations. Provide several options if necessary."
    # question = "describe the image in the image."
    prompt = "USER: <image>\n" + question + "\nASSISTANT:"

    torch.set_default_dtype(torch.float16)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    pipe = pipeline("image-to-text", model=model_class.llava_path, model_kwargs={"quantization_config": quantization_config}) 

    with torch.no_grad():
        img = Image.open('data/remote14/test/13/TopRightBack.png')

        description = pipe(img, prompt=prompt, generate_kwargs={"max_new_tokens": 77})
        # print(pipe.model)
        description = parse_output(description[0]["generated_text"]) 
        print(description)

    description = parse_output(description[0]["generated_text"]) 
    #print(description)

    ###################### images embeds ########################################################################
    inputs = model_class.processor(images=img, return_tensors="pt", do_rescale=False).to(model_class.device)
    embeds = model_class.clip_model.get_image_features(**inputs)
    embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
    image_features = model_class.adapter_image(embeds)
    ##############################################################################################################

    ########### llava descriptions embeds ########################################################################
    descriptions_inputs = model_class.processor(text=description, return_tensors="pt", padding=True, truncation=True, max_length=77).to(model_class.device)
    descriptions_embeds = model_class.clip_model.get_text_features(**descriptions_inputs)
    descriptions_embeds = descriptions_embeds / descriptions_embeds.norm(dim=-1, keepdim=True)
    descriptions_features = model_class.adapter_descriptions(descriptions_embeds)
    ##############################################################################################################

    ###################### questions embeds ######################################################################
    text_inputs = model_class.processor(text=model_class.questions, return_tensors="pt", padding=True).to(model_class.device)
    text_features = model_class.clip_model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    ##############################################################################################################

    ###################### Cosine Similarity fusion ##############################################################
    cos_sim_image = torch.matmul(image_features, text_features.transpose(0, 1))
    #print(f"cos_sim_image: {cos_sim_image}")
    cos_sim_text= torch.matmul(descriptions_features, text_features.transpose(0, 1))
    #print(f"cos_sim_text: {cos_sim_text}")
    cos_sim = cos_sim_image + cos_sim_text
    #print(f"cos_sim: {cos_sim}")
    predicted_classes = torch.argmax(cos_sim, dim=-1)
    ##############################################################################################################

    if plot:
        plt.imshow(img)
        pred_label = model_class.classes[predicted_classes.item()]
        plt.title(pred_label, fontsize=10, color="blue", pad=1)
        plt.text(1, -8, description, ha='center', va='top', fontsize=10, color="black")
        plt.show()

