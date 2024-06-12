import torch
from tqdm import tqdm
from utils.datasets import Remote14
from torch.utils.data import DataLoader

def train_adapter(model_class, epochs, accumulation_steps, train_descriptions, val_descriptions) -> None:
    train_dataset = Remote14(root_dir=model_class.image_dir, is_train=True, descriptions_file=train_descriptions)
    train_loader = DataLoader(train_dataset, batch_size=model_class.bs, shuffle=True, pin_memory=True)
    
    val_dataset = Remote14(root_dir=model_class.image_dir, is_val=True, descriptions_file=val_descriptions)
    val_loader = DataLoader(val_dataset, batch_size=model_class.bs, shuffle=False, pin_memory=True)

    best_val_loss = float('inf')
    model_class.clip_model.eval()  # Freeze the CLIP model
    # torch.nn.utils.clip_grad_norm_(model_class.adapter_image.parameters(), max_norm=1.0)
    # torch.nn.utils.clip_grad_norm_(model_class.adapter_descriptions.parameters(), max_norm=1.0)

    for epoch in range(epochs):
        # Training loop
        model_class.adapter_image.train()
        model_class.adapter_descriptions.train()

        model_class.optimizer_image.zero_grad()
        model_class.optimizer_descriptions.zero_grad()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels, descriptions = batch

            ###################### images embeds ########################################################################
            with torch.no_grad():
                inputs = model_class.processor(images=images, return_tensors="pt", do_rescale=False).to(model_class.device)
                embeds = model_class.clip_model.get_image_features(**inputs)
                embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            image_features = model_class.adapter_image(embeds)
            ##############################################################################################################

            ########### llava descriptions embeds ########################################################################
            with torch.no_grad():
                descriptions_inputs = model_class.processor(text=descriptions, return_tensors="pt", padding=True, truncation=True, max_length=77).to(model_class.device)
                descriptions_embeds = model_class.clip_model.get_text_features(**descriptions_inputs)
                descriptions_embeds = descriptions_embeds / descriptions_embeds.norm(dim=-1, keepdim=True)
            descriptions_features = model_class.adapter_descriptions(descriptions_embeds)
            ##############################################################################################################

            ###################### questions embeds ######################################################################
            with torch.no_grad():
                text_inputs = model_class.processor(text=model_class.questions, return_tensors="pt", padding=True).to(model_class.device)
                text_features = model_class.clip_model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            ##############################################################################################################

            ###################### Cosine Similarity fusion ##############################################################
            cos_sim_image = torch.matmul(image_features, text_features.transpose(0, 1))
            cos_sim_text= torch.matmul(descriptions_features, text_features.transpose(0, 1))
            cos_sim = cos_sim_image + cos_sim_text
            predicted_classes = torch.argmax(cos_sim, dim=-1)
            ##############################################################################################################

            loss = model_class.criterion(cos_sim, labels.to(model_class.device))

            train_loss = loss.item()
            loss.backward()

            if (epoch + 1) % accumulation_steps == 0:  # Wait for several mini-batches
                model_class.optimizer_image.step()
                model_class.optimizer_descriptions.step()

                model_class.scheduler_image.step()
                model_class.scheduler_descriptions.step()

                model_class.optimizer_image.zero_grad()
                model_class.optimizer_descriptions.zero_grad()

            # print(f"Lr image adapter: {model_class.optimizer_image.param_groups[0]['lr']}")
            # print(f"Lr descriptions adapter: {model_class.optimizer_descriptions.param_groups[0]['lr']}")
            # for name, param in self.adapter_image.named_parameters():
            #     print(f'Parameter: {name}, Grad: {param.grad}, Value: {param.data}')

            model_class.metrics['train']['preds'].extend(predicted_classes.cpu().tolist())
            model_class.metrics['train']['gts'].extend(labels.cpu().tolist())

        torch.cuda.empty_cache()
        # Validation loop
        model_class.adapter_image.eval() 
        model_class.adapter_descriptions.eval() 
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch+1}/{epochs}"):
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
                cos_sim = torch.matmul(image_features, text_features.transpose(0, 1))
                cos_sim += torch.matmul(descriptions_features, text_features.transpose(0, 1))
                predicted_classes = torch.argmax(cos_sim, dim=-1)
                ##############################################################################################################

                model_class.metrics['val']['preds'].extend(predicted_classes.cpu().tolist())
                model_class.metrics['val']['gts'].extend(labels.cpu().tolist())

                val_loss = model_class.criterion(cos_sim, labels.to(model_class.device))
                val_losses.append(val_loss.item())

        train_acc = sum([1 for gt, pred in zip(model_class.metrics['train']['gts'], model_class.metrics['train']['preds']) if gt == pred]) / len(model_class.metrics['train']['gts'])
        val_acc = sum([1 for gt, pred in zip(model_class.metrics['val']['gts'], model_class.metrics['val']['preds']) if gt == pred]) / len(model_class.metrics['val']['gts'])
        model_class.writer.add_scalars('Acc', {'Train': train_acc, 'Validation': val_acc}, epoch)

        mean_val_loss = sum(val_losses) / len(val_losses)
        # print(f"val_losses: {val_losses}")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(model_class.adapter_image.state_dict(), model_class.save_path)
            torch.save(model_class.adapter_descriptions.state_dict(), model_class.save_path_descriptions)
            print(f"Model saved at epoch {epoch+1}, with validation loss: {mean_val_loss}, path: {model_class.save_path, model_class.save_path_descriptions}, train_acc: {train_acc}, val_acc: {val_acc}")

        model_class.writer.add_scalars('Losses', {'Train': train_loss, 'Val': mean_val_loss}, epoch)
        model_class.metrics = model_class._reset_metrics()
