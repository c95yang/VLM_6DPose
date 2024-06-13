import os
from PIL import Image 
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
import json
from torchvision import transforms
from matplotlib import pyplot as plt

class Remote14(Dataset):
    def __init__(self, root_dir, descriptions_file=None, is_test=False, is_val=False, is_train=False):
        self.is_test = is_test
        self.is_val = is_val
        self.is_train = is_train
        self.root_dir = root_dir
        self.image_paths = []
        self.desired_size = (800, 800)

        self.descriptions_file = descriptions_file
        if self.descriptions_file is not None:
            self.descriptions = self.load_descriptions(self.descriptions_file)

        self.labels = []
        self.classes = ['back', 'bottom', 'bottomleftback', 'bottomleftfront', 
                        'bottomrightback', 'bottomrightfront', 'front', 'left', 
                        'right', 'top', 'topleftback', 'topleftfront', 'toprightback', 'toprightfront']
        self.load_images_and_labels()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        # print(self.class_to_idx)

        self.transform = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ], p=0.5),  

            transforms.RandomApply([
                transforms.RandomResizedCrop(size=self.desired_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
            ], p=0.2), 

            transforms.Resize(self.desired_size), 
        ])

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
                    if  path == os.path.join(self.root_dir, 'test') and img.size != self.desired_size:
                        img = img.resize(self.desired_size, Image.ANTIALIAS)
                        img.save(img_path)
                    img.close()

                    self.image_paths.append(img_path)
                    self.labels.append(label)

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
        image = self.transform(image)
        # plt.imshow(image)
        # plt.show()
        image = transforms.ToTensor()(image)

        label_idx = self.class_to_idx[label]

        if self.descriptions_file is not None:
            description = self.descriptions[img_path]
            return image, label_idx, description
        else:
            return image, label_idx


    def get_class_labels(self):
        return self.classes
    
    def get_all_image_paths(self):
        return self.image_paths
    
    def load_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        image = transforms.ToTensor()(image)
        return image