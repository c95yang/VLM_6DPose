import torch
from tqdm import tqdm
from utils.datasets import Remote60
from torch.utils.data import DataLoader
from utils.misc import HammingLoss, hamming_dist, log_memory_usage, interpolate_color
import matplotlib.pyplot as plt
import textwrap
from utils.positions import classes

def feature_select(image_forward_outs):
    image_features = image_forward_outs.hidden_states[-1]
    image_features = image_features[:, 1:]
    return image_features

def train(model_class, epochs, train_descriptions, val_descriptions, lam, zeroshot) -> None:
    fusion = model_class.fusion
    train_dataset = Remote60(root_dir=model_class.image_dir, is_train=True, descriptions_file=train_descriptions)
    train_loader = DataLoader(train_dataset, batch_size=model_class.bs, pin_memory=True, num_workers=2) 
    
    val_dataset = Remote60(root_dir=model_class.image_dir, is_val=True, descriptions_file=val_descriptions)
    val_loader = DataLoader(val_dataset, batch_size=model_class.bs, pin_memory=True, num_workers=2, shuffle=False)

    best_val_loss = float('inf')
    if model_class.embedder == 'clip':
        model_class.clip_model.eval()  
    elif model_class.embedder == 'blip':
        model_class.blip_model.eval()

    for epoch in range(epochs):
        # Training loop
        model_class.adapter_image.train()
        model_class.optimizer_image.zero_grad()

        if fusion:
            model_class.adapter_descriptions.train()
            model_class.optimizer_descriptions.zero_grad()

        with torch.no_grad():
            if model_class.embedder == 'clip':
                text_inputs = model_class.clip_processor(text=model_class.questions, return_tensors="pt", padding=True).to(model_class.device)
                text_features = model_class.clip_model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(model_class.device)
            elif model_class.embedder == 'blip':
                text_features = model_class.blip_model(images, model_class.questions, mode='text')[:,0]
                text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(model_class.device)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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

            # ###################### blip multimodal ########################################################################
            # with torch.no_grad():
            #     images = images.to(model_class.device)
            #     multimodal_features = model_class.blip_model(images, descriptions, mode='multimodal')
            #     multimodal_features = multimodal_features.mean(dim=1)
            #     multimodal_features = multimodal_features / multimodal_features.norm(p=2, dim=-1, keepdim=True).to(model_class.device)
            # multimodal_features = model_class.adapter_image(multimodal_features)

            # cos_sim = torch.nn.functional.cosine_similarity(multimodal_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
            # print(cos_sim)
            # print(torch.nn.Softmax(dim=1)(cos_sim))
            # predicted_classes = torch.argmax(cos_sim, dim=-1)
            # ##############################################################################################################

            ###################### llava approach ##############################################################
            # image_tokens = model_class.mm_projector(image_features[0])

            # messages = [ 
            #     {"role": "system", "content": "You are a helpful AI assistant."}, 
            #     {"role": "user", "content": "Describe the romote in the image."}, 
            #     {"role": "assistant", "content": descriptions[0]},
            #     {"role": "user", "content": "What is the orientation of the remote?"}, 
            # ] 

            # input_text = ""
            # for message in messages:
            #     input_text += f"{message['role']}: {message['content']}\n"

            # input_ids = model_class.phi3tokenizer(input_text, return_tensors="pt").to("cuda")
            # print(input_ids)

            # generation_args = {
            #     "max_new_tokens": 100,
            #     "temperature": 0.0,
            #     "do_sample": False,
            # }

            # input_ids = torch.cat((input_ids, image_tokens),dim=1)
            # print(input_ids)

            # with torch.no_grad():
            #     output_ids = model_class.phi3model.generate(input_ids, **generation_args)

            # print(output_ids)

            # output_text = model_class.phi3tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # print(output_text[len(input_text):]) 

            # # print(model_class.phi3model.decoder.layers[-1].final_layer_norm.weight.dtype)

            # # cos_sim = torch.nn.functional.cosine_similarity(tokens.unsqueeze(1), text_features.unsqueeze(0), dim=2)
            # tokens = torch.nn.functional.normalize(tokens, p=2, dim=-1)
            # text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
            # cos_sim = torch.matmul(tokens, text_features.transpose(0, 1))
            # print(cos_sim)
            # print(torch.nn.Softmax(dim=1)(cos_sim))
            # predicted_classes = torch.argmax(cos_sim, dim=-1)  
            ##############################################################################################################

            loss = model_class.criterion(cos_sim, labels.to(model_class.device))
            # val_loss = model_class.criterion(predicted_classes, labels)
            train_loss = loss.item()

            if not zeroshot:
                loss.backward()

                model_class.optimizer_image.step()
                model_class.scheduler_image.step()
                model_class.optimizer_image.zero_grad()

                if fusion:
                    model_class.optimizer_descriptions.step()
                    model_class.scheduler_descriptions.step()
                    model_class.optimizer_descriptions.zero_grad()

            model_class.metrics['train']['preds'].extend(predicted_classes.cpu().tolist())
            model_class.metrics['train']['gts'].extend(labels.cpu().tolist())

        # Validation loop
        model_class.adapter_image.eval() 
        if fusion:
            model_class.adapter_descriptions.eval() 
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch+1}/{epochs}"):
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

                val_loss = model_class.criterion(cos_sim, labels.to(model_class.device))
                # val_loss = model_class.criterion(predicted_classes, labels)
                val_losses.append(val_loss.item())

                model_class.metrics['val']['preds'].extend(predicted_classes.cpu().tolist())
                model_class.metrics['val']['gts'].extend(labels.cpu().tolist())

                # plot = (epoch + 1) % 100 == 0
                # if plot:
                #     num_images = len(images) 
                #     num_rows = 2
                #     num_cols = 8

                #     fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(30, 30))  
                #     for ax_row in axes:
                #         for ax in ax_row:
                #             ax.axis('off')

                #     for i in range(num_images):
                #         img = images[i].cpu().numpy().transpose((1, 2, 0))
                #         pred = predicted_classes[i]
                #         gt = labels[i]

                #         pred_class = classes[pred]  
                #         gt_class = classes[gt]  

                #         row = i // num_cols  
                #         col = i % num_cols  
                #         ax = axes[row, col]  
                #         ax.imshow(img)

                #         hamming_distance = hamming_dist(pred, gt).item()
                #         color = interpolate_color(hamming_distance)

                #         ax.set_title(gt_class, fontsize=15, color="black", pad=1)
                #         ax.text(0.5, -0.02, pred_class, transform=ax.transAxes, ha='center', va='top', fontsize=15, color=color)
                #         wrapped_text = "\n".join(textwrap.wrap(f"d = {hamming_dist(pred, gt)}", width=25))
                #         ax.text(0.5, -0.2, wrapped_text, transform=ax.transAxes, ha='center', va='top', fontsize=15, color=color)

                #     plt.show()

        train_acc = sum([1 for gt, pred in zip(model_class.metrics['train']['gts'], model_class.metrics['train']['preds']) if gt == pred]) / len(model_class.metrics['train']['gts'])
        val_acc = sum([1 for gt, pred in zip(model_class.metrics['val']['gts'], model_class.metrics['val']['preds']) if gt == pred]) / len(model_class.metrics['val']['gts'])
        model_class.writer.add_scalars('Acc_3', {'Train': train_acc, 'Validation': val_acc}, epoch)

        train_acc_h2 = sum([1 for gt, pred in zip(model_class.metrics['train']['gts'], model_class.metrics['train']['preds']) if hamming_dist(gt, pred) <= 1]) / len(model_class.metrics['train']['gts'])
        val_acc_h2 = sum([1 for gt, pred in zip(model_class.metrics['val']['gts'], model_class.metrics['val']['preds']) if hamming_dist(gt, pred) <= 1]) / len(model_class.metrics['val']['gts'])
        model_class.writer.add_scalars('Acc_2', {'Train': train_acc_h2, 'Validation': val_acc_h2}, epoch)

        train_acc_h1 = sum([1 for gt, pred in zip(model_class.metrics['train']['gts'], model_class.metrics['train']['preds']) if hamming_dist(gt, pred) <= 2]) / len(model_class.metrics['train']['gts'])
        val_acc_h1 = sum([1 for gt, pred in zip(model_class.metrics['val']['gts'], model_class.metrics['val']['preds']) if hamming_dist(gt, pred) <= 2]) / len(model_class.metrics['val']['gts'])
        model_class.writer.add_scalars('Acc_1', {'Train': train_acc_h1, 'Validation': val_acc_h1}, epoch)

        mean_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch+1}, validation loss: {mean_val_loss}, train_acc: {train_acc}, val_acc: {val_acc}")


        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(model_class.adapter_image.state_dict(), model_class.save_path)
            if fusion:
                torch.save(model_class.adapter_descriptions.state_dict(), model_class.save_path_descriptions)
                print(f"Model saved at epoch {epoch+1}, with path: {model_class.save_path, model_class.save_path_descriptions}")
            else:
                print(f"Model saved at epoch {epoch+1}, with path: {model_class.save_path}")

        model_class.writer.add_scalars('Losses', {'Train': train_loss, 'Val': mean_val_loss}, epoch)
        model_class.metrics = model_class._reset_metrics()
