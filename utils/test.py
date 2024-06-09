import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from utils.datasets import Remote14
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from transformers import BitsAndBytesConfig, pipeline

from utils.misc import parse_output

def test_adapter(model_class, split: str = 'val') -> None:
        if split == 'val':
            val_dataset = Remote14(root_dir=model_class.image_dir, is_val=True)
            dataloader = DataLoader(val_dataset, batch_size=model_class.bs, shuffle=False, pin_memory=True)
        elif split == 'test':
            test_dataset = Remote14(root_dir=model_class.image_dir, is_test=True)
            dataloader = DataLoader(test_dataset, batch_size=model_class.bs, shuffle=False, pin_memory=True)
        else:
            train_dataset = Remote14(root_dir=model_class.image_dir, is_train=True)
            dataloader = DataLoader(train_dataset, batch_size=model_class.bs, shuffle=True, pin_memory=True)
            
        question = "There is a remote in the image. Describe what you see and tell me from which direction is the photo taken."
        prompt = "USER: <image>\n" + question + "\nASSISTANT:"

        topil = ToPILImage()
        llava_path = "llava-hf/llava-1.5-7b-hf"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        pipe = pipeline("image-to-text", model=llava_path, model_kwargs={"quantization_config": quantization_config})

        model_class.model.eval()

        model_class.adapter_image.eval()
        model_class.adapter_image.load_state_dict(torch.load(model_class.load_path))
        model_class.adapter_descriptions.eval()
        model_class.adapter_descriptions.load_state_dict(torch.load(model_class.load_path_descriptions))

        for batch in dataloader:
            images, labels = batch

            ###################### images embeds ########################################################################
            inputs = model_class.processor(images=images, return_tensors="pt", do_rescale=False).to(model_class.device)
            embeds = model_class.model.get_image_features(**inputs)
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            image_features = model_class.adapter_image(embeds)
            ##############################################################################################################

            ########### llava descriptions embeds ########################################################################
            images = [topil(image) for image in images]
            descriptions = pipe(images, prompt=prompt, generate_kwargs={"max_new_tokens": 77})
            descriptions = [parse_output(description[0]["generated_text"]) for description in descriptions]
            # print("descriptions: ", descriptions)
            descriptions_inputs = model_class.processor(text=descriptions, return_tensors="pt", padding=True, truncation=True, max_length=77).to(model_class.device)
            descriptions_embeds = model_class.model.get_text_features(**descriptions_inputs)
            descriptions_embeds = descriptions_embeds / descriptions_embeds.norm(dim=-1, keepdim=True)
            descriptions_features = model_class.adapter_descriptions(descriptions_embeds)
            ##############################################################################################################

            ###################### questions embeds ######################################################################
            text_inputs = model_class.processor(text=model_class.questions, return_tensors="pt", padding=True).to(model_class.device)
            text_features = model_class.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            ##############################################################################################################

            ###################### Cosine Similarity fusion ##############################################################
            cos_sim = torch.matmul(image_features, text_features.transpose(0, 1))
            cos_sim += torch.matmul(descriptions_features, text_features.transpose(0, 1))
            predicted_classes = torch.argmax(cos_sim, dim=-1)
            ##############################################################################################################
            # print(cos_sim)     
            # print("labels: ", labels) 
            # print("predicted_classes:", predicted_classes)

            model_class.metrics[split]['preds'].extend(predicted_classes.tolist())
            model_class.metrics[split]['gts'].extend(labels.cpu().tolist())


            acc = sum([1 for gt, pred in zip(model_class.metrics[split]['gts'], model_class.metrics[split]['preds']) if gt == pred]) / len(model_class.metrics[split]['gts'])
            print(f"Accuracy: {acc}")
            print(model_class.metrics)
            
            num_images = len(images) 
            num_rows = 1
            num_cols = 8

            fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 20))  
            for ax_row in axes:
                for ax in ax_row:
                    ax.axis('off')

            for i in range(num_images):
                img = images[i].cpu().numpy().transpose((1, 2, 0))

                pred_label = model_class.classes[predicted_classes[i]]
                gt_label = model_class.classes[labels[i].cpu().item()]  

                row = i // num_cols  
                col = i % num_cols  
                ax = axes[row, col]  
                ax.imshow(img)
                ax.set_title(pred_label, fontsize=10, color="green" if pred_label == gt_label else "red", pad=1)
                ax.text(0.5, -0.02, gt_label, transform=ax.transAxes, ha='center', va='top', fontsize=10, color="black")

            plt.show()

        print(classification_report(model_class.metrics[split]['gts'], model_class.metrics[split]['preds'], target_names=model_class.classes))
        model_class._reset_metrics()
