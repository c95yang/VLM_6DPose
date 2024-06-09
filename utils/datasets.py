import os
from PIL import Image as Image
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
import json

class Remote14(Dataset):
    def __init__(self, root_dir, descriptions_file, is_test=False, is_val=False, is_train=False):
        self.is_test = is_test
        self.is_val = is_val
        self.is_train = is_train
        self.root_dir = root_dir
        self.image_paths = []

        self.descriptions_file = descriptions_file
        self.descriptions = self.load_descriptions(self.descriptions_file)

        self.labels = []
        self.classes = ['back', 'bottom', 'bottomleftback', 'bottomleftfront', 
                        'bottomrightback', 'bottomrightfront', 'front', 'left', 
                        'right', 'top', 'topleftback', 'topleftfront', 'toprightback', 'toprightfront']
        self.load_images_and_labels()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        # print(self.class_to_idx)
        self.desired_size = (800, 800)
    
    def load_images_and_labels(self): 
        if self.is_val:
            path = os.path.join(self.root_dir, 'val')
        elif self.is_test:
            path = os.path.join(self.root_dir, 'test')
        elif self.is_train:
            path = os.path.join(self.root_dir, 'train')
        else:
            print("Error: Invalid dataset split")
            
        for subdir, _, files in os.walk(path):
            for file in files:
                if file.endswith('.png'):
                    img_path = os.path.join(subdir, file)
                    label = os.path.splitext(file)[0].lower()

                    img = Image.open(img_path)
                    if  path == os.path.join(self.root_dir, 'test') and img.size != (800, 800):
                        img = img.resize((800, 800), Image.ANTIALIAS)
                        img.save(img_path)
                    img.close()

                    self.image_paths.append(img_path)
                    self.labels.append(label)

                    # if path == os.path.join(self.root_dir, 'test'):
                    #     if label == 'top' or label == 'bottom':
                    #         base_name = os.path.splitext(file)[0]
                    #         ext = os.path.splitext(file)[1]

                    #         image = Image.open(img_path)
                    #         for angle in [45, 90, 135, 180, 225, 270, 315]:
                    #             # rotated_image = image.rotate(angle, fillcolor=(187, 187, 187))
                    #             rotated_image = image.rotate(angle, fillcolor=(255, 255, 255))
                    #             rotated_file = f"{base_name}_rot{angle}{ext}"
                    #             rotated_path = os.path.join(subdir, rotated_file)
                    #             rotated_image.save(rotated_path)
                    #             print(rotated_path)

                    #             self.image_paths.append(rotated_path)
                    #             self.labels.append(label)

    def load_descriptions(self, descriptions_file):
        with open(descriptions_file, "r") as f:
            descriptions = json.load(f)
        return descriptions
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        label = self.labels[idx]
        if '_rot' in label:
            label = label.split('_rot')[0]

        image = Image.open(img_path).convert("RGB")
        image = F.to_tensor(image) 
        label_idx = self.class_to_idx[label]
        description = self.descriptions[img_path]
        return image, label_idx, description

    def get_class_labels(self):
        return self.classes
    
    def get_all_image_paths(self):
        return self.image_paths
    
    def load_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        image = F.to_tensor(image)
        return image
    

# class Remote14_raw(Dataset):
#     def __init__(self, root_dir, is_test=False, is_val=False, is_train=False):
#         self.is_test = is_test
#         self.is_val = is_val
#         self.is_train = is_train
#         self.root_dir = root_dir
#         self.image_paths = []
#         self.labels = []
#         self.classes = ['back', 'bottom', 'bottomleftback', 'bottomleftfront', 
#                         'bottomrightback', 'bottomrightfront', 'front', 'left', 
#                         'right', 'top', 'topleftback', 'topleftfront', 'toprightback', 'toprightfront']
#         self.load_images_and_labels()
#         self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
#         # print(self.class_to_idx)
#         self.desired_size = (800, 800)
    
#     def load_images_and_labels(self): 
#         if self.is_val:
#             path = os.path.join(self.root_dir, 'val')
#         elif self.is_test:
#             path = os.path.join(self.root_dir, 'test')
#         elif self.is_train:
#             path = os.path.join(self.root_dir, 'train')
#         else:
#             #error
#             print("Error: Invalid dataset split")
            
#         for subdir, _, files in os.walk(path):
#             for file in files:
#                 if file.endswith('.png'):
#                     img_path = os.path.join(subdir, file)
#                     label = os.path.splitext(file)[0].lower()

#                     img = Image.open(img_path)
#                     if  path == os.path.join(self.root_dir, 'test') and img.size != (800, 800):
#                         img = img.resize((800, 800), Image.ANTIALIAS)
#                         img.save(img_path)
#                     img.close()

#                     self.image_paths.append(img_path)
#                     self.labels.append(label)
    
#     def __len__(self):
#         return len(self.image_paths)
    
#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]

#         label = self.labels[idx]
#         if '_rot' in label:
#             label = label.split('_rot')[0]

#         image = Image.open(img_path).convert("RGBA")
#         image = F.to_tensor(image)
#         label_idx = self.class_to_idx[label]
#         return image, label_idx

#     def get_class_labels(self):
#         return self.classes
    