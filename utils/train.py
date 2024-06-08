import torch
from tqdm import tqdm
from utils.datasets import Remote14
from torch.utils.data import DataLoader

def train_adapter(model_class, epochs) -> None:
    train_dataset = Remote14(root_dir=model_class.image_dir, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=model_class.bs, shuffle=True, pin_memory=True)
    
    val_dataset = Remote14(root_dir=model_class.image_dir, is_val=True)
    val_loader = DataLoader(val_dataset, batch_size=model_class.bs, shuffle=False, pin_memory=True)

    model_class.model.eval()  # Freeze the CLIP model
    model_class.adapter.train()

    best_val_loss = float('inf')  

    for epoch in range(epochs):
        # Training loop
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = batch
            #print("images: ", images)

            inputs = model_class.processor(images=images, return_tensors="pt", do_rescale=False).to(model_class.device)

            embeds = model_class.model.get_image_features(**inputs)
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)

            # print("embeds: ", embeds)

            image_features = model_class.adapter(embeds)
            # print("image_features: ", image_features)

            text_inputs = model_class.processor(text=model_class.questions, return_tensors="pt", padding=True).to(model_class.device)
            with torch.no_grad():
                text_features = model_class.model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            cos_sim = torch.matmul(image_features, text_features.transpose(0, 1))
            predicted_classes = torch.argmax(cos_sim, dim=-1)
            # print(cos_sim)     
            # print("labels: ", labels) 
            # print("predicted_classes:", predicted_classes)

            loss = model_class.criterion(cos_sim, labels.to(model_class.device))

            # print(f"Train_Loss: {loss}")

            train_loss = loss.item()
            model_class.optimizer.zero_grad()
            loss.backward()

            model_class.optimizer.step()
            model_class.scheduler.step()

            print(f"Lr: {model_class.optimizer.param_groups[0]['lr']}")
            # for name, param in self.adapter.named_parameters():
            #     print(f'Parameter: {name}, Grad: {param.grad}, Value: {param.data}')

            model_class.metrics['train']['preds'].extend(predicted_classes.cpu().tolist())
            model_class.metrics['train']['gts'].extend(labels.cpu().tolist())

        # Validation loop
        model_class.adapter.eval()  
        with torch.no_grad():
            val_losses = []
            for batch in tqdm(val_loader, desc=f"Validation {epoch+1}/{epochs}"):
                images, labels = batch
                images, labels = images.to(model_class.device), labels.to(model_class.device)

                inputs = model_class.processor(images=images, return_tensors="pt", do_rescale=False).to(model_class.device)

                embeds = model_class.model.get_image_features(**inputs)
                embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
                image_features = model_class.adapter(embeds)

                text_inputs = model_class.processor(text=model_class.questions, return_tensors="pt", padding=True).to(model_class.device)
                with torch.no_grad():
                    text_features = model_class.model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                cos_sim = torch.matmul(image_features, text_features.transpose(0, 1))
                # print("cos_sim: ", cos_sim)

                predicted_classes = torch.argmax(cos_sim, dim=-1)
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
            torch.save(model_class.adapter.state_dict(), model_class.save_path)
            print(f"Model saved at epoch {epoch+1}, with validation loss: {mean_val_loss}, path: {model_class.save_path}, train_acc: {train_acc}, val_acc: {val_acc}")

        model_class.writer.add_scalars('Losses', {'Train': train_loss, 'Val': mean_val_loss}, epoch)
        model_class.metrics = model_class._reset_metrics()
