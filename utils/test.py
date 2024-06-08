import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from utils.datasets import Remote14
from torch.utils.data import DataLoader

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

        model_class.adapter.eval()
        model_class.adapter.load_state_dict(torch.load(model_class.load_path))

        for batch in dataloader:
            images, labels = batch

            inputs = model_class.processor(images=images, return_tensors="pt", do_rescale=False)
            inputs.to(model_class.model.device)
            
            embeds = model_class.model.get_image_features(**inputs)
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            image_features = model_class.adapter(embeds) 

            text_inputs = model_class.processor(text=model_class.questions, return_tensors="pt", padding=True).to(model_class.device)
            text_features = model_class.model.get_text_features(**text_inputs) 

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarities = torch.matmul(image_features, text_features.transpose(0, 1)) 
            norm_text_features = torch.norm(text_features, p=2, dim=-1, keepdim=True)
            norm_image_features = torch.norm(image_features, p=2, dim=-1, keepdim=True)
            cos_sim = torch.div(similarities, torch.matmul(norm_image_features, norm_text_features.transpose(0, 1))) 

            predicted_classes = torch.argmax(cos_sim, dim=-1)
            # print(cos_sim)     
            # print("labels: ", labels) 
            # print("predicted_classes:", predicted_classes)

            model_class.metrics[split]['preds'].extend(predicted_classes.tolist())
            model_class.metrics[split]['gts'].extend(labels.cpu().tolist())


            acc = sum([1 for gt, pred in zip(model_class.metrics[split]['gts'], model_class.metrics[split]['preds']) if gt == pred]) / len(model_class.metrics[split]['gts'])
            print(f"Accuracy: {acc}")
            print(model_class.metrics)
            
            num_images = len(images) 
            num_rows = 4
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
