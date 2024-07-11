import torch
from tqdm import tqdm
from utils.datasets import Remote14
from torch.utils.data import DataLoader

def train_adapter(model_class, epochs, train_descriptions, val_descriptions, fusion, lam) -> None:
    train_dataset = Remote14(root_dir=model_class.image_dir, is_train=True, descriptions_file=train_descriptions)
    train_loader = DataLoader(train_dataset, batch_size=model_class.bs, pin_memory=True, num_workers=2) 
    
    val_dataset = Remote14(root_dir=model_class.image_dir, is_val=True, descriptions_file=val_descriptions)
    val_loader = DataLoader(val_dataset, batch_size=model_class.bs, pin_memory=True, num_workers=2)

    best_val_loss = float('inf')
    # model_class.clip_model.eval()  
    model_class.blip_model.eval()

    for epoch in range(epochs):
        # Training loop
        model_class.adapter_image.train()
        model_class.adapter_descriptions.train()

        model_class.optimizer_image.zero_grad()
        model_class.optimizer_descriptions.zero_grad()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels, descriptions = batch

            # ###################### clip embeds ########################################################################
            # with torch.no_grad():
            #     inputs = model_class.processor(images=images, return_tensors="pt", do_rescale=False).to(model_class.device)
            #     embeds = model_class.clip_model.get_image_features(**inputs)
            #     embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)

            # image_features = model_class.adapter_image(embeds)

            # if fusion:
            #     with torch.no_grad():
            #         descriptions_inputs = model_class.processor(text=descriptions, return_tensors="pt", padding=True, truncation=True, max_length=77).to(model_class.device)
            #         descriptions_embeds = model_class.clip_model.get_text_features(**descriptions_inputs)
            #         descriptions_embeds = descriptions_embeds / descriptions_embeds.norm(dim=-1, keepdim=True)
            #     descriptions_features = model_class.adapter_descriptions(descriptions_embeds)

            # with torch.no_grad():
            #     text_inputs = model_class.processor(text=model_class.questions, return_tensors="pt", padding=True).to(model_class.device)
            #     text_features = model_class.clip_model.get_text_features(**text_inputs)
            #     text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(model_class.device)
            # ##############################################################################################################

            ###################### blip embeds ########################################################################
            with torch.no_grad():
                images = images.to(model_class.device)
                image_features = model_class.blip_model(images, descriptions, mode='image')
                image_features = image_features.mean(dim=1)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True).to(model_class.device)
            image_features = model_class.adapter_image(image_features)

            if fusion:
                with torch.no_grad():
                    descriptions_features = model_class.blip_model(images, descriptions, mode='text')
                    descriptions_features = descriptions_features.mean(dim=1)
                    descriptions_features = descriptions_features / descriptions_features.norm(p=2, dim=-1, keepdim=True).to(model_class.device)
                descriptions_features = model_class.adapter_descriptions(descriptions_features)

            with torch.no_grad():
                text_features = model_class.blip_model(images, model_class.questions, mode='text')[:,0]
                text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(model_class.device)
            ##############################################################################################################

            ###################### Cosine Similarity fusion ##############################################################
            cos_sim = torch.nn.functional.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
            print(cos_sim)
            print(torch.nn.Softmax(dim=1)(cos_sim))
            if fusion:
                cos_sim += lam * torch.nn.functional.cosine_similarity(descriptions_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
            predicted_classes = torch.argmax(cos_sim, dim=-1)
            ##############################################################################################################

            # ###################### blip multimodal ########################################################################
            # with torch.no_grad():
            #     images = images.to(model_class.device)
            #     multimodal_features = model_class.blip_model(images, descriptions, mode='multimodal')
            #     multimodal_features = multimodal_features.mean(dim=1)
            #     multimodal_features = multimodal_features / multimodal_features.norm(p=2, dim=-1, keepdim=True).to(model_class.device)
            # multimodal_features = model_class.adapter_image(multimodal_features)

            # with torch.no_grad():
            #     text_features = model_class.blip_model(images, model_class.questions, mode='text')
            #     text_features = text_features.mean(dim=1)
            #     text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(model_class.device)

            # cos_sim = torch.nn.functional.cosine_similarity(multimodal_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
            # print(cos_sim)
            # print(torch.nn.Softmax(dim=1)(cos_sim))
            # predicted_classes = torch.argmax(cos_sim, dim=-1)
            # ##############################################################################################################

            loss = model_class.criterion(cos_sim, labels.to(model_class.device))

            train_loss = loss.item()
            # print(f"Train loss: {train_loss}")
            loss.backward()

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
                
                # ###################### clip embeds ########################################################################
                # with torch.no_grad():
                #     inputs = model_class.processor(images=images, return_tensors="pt", do_rescale=False).to(model_class.device)
                #     embeds = model_class.clip_model.get_image_features(**inputs)
                #     embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)

                # image_features = model_class.adapter_image(embeds)

                # if fusion:
                #     with torch.no_grad():
                #         descriptions_inputs = model_class.processor(text=descriptions, return_tensors="pt", padding=True, truncation=True, max_length=77).to(model_class.device)
                #         descriptions_embeds = model_class.clip_model.get_text_features(**descriptions_inputs)
                #         descriptions_embeds = descriptions_embeds / descriptions_embeds.norm(dim=-1, keepdim=True)
                #     descriptions_features = model_class.adapter_descriptions(descriptions_embeds)

                # with torch.no_grad():
                #     text_inputs = model_class.processor(text=model_class.questions, return_tensors="pt", padding=True).to(model_class.device)
                #     text_features = model_class.clip_model.get_text_features(**text_inputs)
                #     text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(model_class.device)
                # ##############################################################################################################

                ###################### blip embeds ########################################################################
                with torch.no_grad():
                    images = images.to(model_class.device)
                    image_features = model_class.blip_model(images, descriptions, mode='image')
                    image_features = image_features.mean(dim=1)
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True).to(model_class.device)
                image_features = model_class.adapter_image(image_features)

                if fusion:
                    with torch.no_grad():
                        descriptions_features = model_class.blip_model(images, descriptions, mode='text')
                        descriptions_features = descriptions_features.mean(dim=1)
                        descriptions_features = descriptions_features / descriptions_features.norm(p=2, dim=-1, keepdim=True).to(model_class.device)
                    descriptions_features = model_class.adapter_descriptions(descriptions_features)

                with torch.no_grad():
                    text_features = model_class.blip_model(images, model_class.questions, mode='text')[:,0]
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(model_class.device)
                ##############################################################################################################

                ###################### Cosine Similarity fusion ##############################################################
                cos_sim = torch.nn.functional.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
                if fusion:
                    cos_sim += lam * torch.nn.functional.cosine_similarity(descriptions_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
                predicted_classes = torch.argmax(cos_sim, dim=-1)
                ##############################################################################################################

                # ###################### blip multimodal ########################################################################
                # with torch.no_grad():
                #     images = images.to(model_class.device)
                #     multimodal_features = model_class.blip_model(images, descriptions, mode='multimodal')[:,0]
                #     multimodal_features = multimodal_features / multimodal_features.norm(p=2, dim=-1, keepdim=True).to(model_class.device)
                # multimodal_features = model_class.adapter_image(multimodal_features)

                # with torch.no_grad():
                #     text_features = model_class.blip_model(images, model_class.questions, mode='text')[:,0]
                #     text_features = text_features / text_features.norm(dim=-1, keepdim=True).to(model_class.device)

                # # cos_sim = torch.matmul(multimodal_features, text_features.transpose(0, 1))
                # cos_sim = torch.nn.functional.cosine_similarity(multimodal_features.unsqueeze(1), text_features.unsqueeze(0), dim=2)
                # predicted_classes = torch.argmax(cos_sim, dim=-1)
                # ##############################################################################################################

                model_class.metrics['val']['preds'].extend(predicted_classes.cpu().tolist())
                model_class.metrics['val']['gts'].extend(labels.cpu().tolist())

                val_loss = model_class.criterion(cos_sim, labels.to(model_class.device))
                val_losses.append(val_loss.item())

        train_acc = sum([1 for gt, pred in zip(model_class.metrics['train']['gts'], model_class.metrics['train']['preds']) if gt == pred]) / len(model_class.metrics['train']['gts'])
        val_acc = sum([1 for gt, pred in zip(model_class.metrics['val']['gts'], model_class.metrics['val']['preds']) if gt == pred]) / len(model_class.metrics['val']['gts'])
        model_class.writer.add_scalars('Acc', {'Train': train_acc, 'Validation': val_acc}, epoch)

        mean_val_loss = sum(val_losses) / len(val_losses)
        print(f"Epoch {epoch+1}, validation loss: {mean_val_loss}, train_acc: {train_acc}, val_acc: {val_acc}")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(model_class.adapter_image.state_dict(), model_class.save_path)
            torch.save(model_class.adapter_descriptions.state_dict(), model_class.save_path_descriptions)
            print(f"Model saved at epoch {epoch+1}, with path: {model_class.save_path, model_class.save_path_descriptions}")

        model_class.writer.add_scalars('Losses', {'Train': train_loss, 'Val': mean_val_loss}, epoch)
        model_class.metrics = model_class._reset_metrics()
