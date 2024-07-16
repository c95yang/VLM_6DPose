import os
from PIL import Image 
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
import json
from torchvision import transforms
from matplotlib import pyplot as plt
from torch.utils.data import IterableDataset

def assign_label(filename):
    if filename == "render_position((0.40784254248348, -0.27499999999999997, -0.08966303887673437))_rotation(<Euler (x=1.7511, y=-0.0000, z=0.9775), order='XYZ'>).png":
        return "bottomrightfront"
    elif filename == "render_position((0.244377030147626, -0.09166666666666665, 0.42647050233099176))_rotation(<Euler (x=0.5492, y=-0.0000, z=1.2119), order='XYZ'>).png":
        return "topfrontright"
    elif filename == "render_position((0.1178017771025154, -0.3083333333333333, 0.3755706283337994))_rotation(<Euler (x=0.7210, y=-0.0000, z=0.3649), order='XYZ'>).png":
        return "topright"
    elif filename == "render_position((0.1521582080309768, 0.2416666666666666, -0.4104206402595077))_rotation(<Euler (x=-0.6079, y=-3.1416, z=-0.5619), order='XYZ'>).png":
        return "bottomleft"
    elif filename == "render_position((0.2630477194325678, 0.425, 0.013449806739322445))_rotation(<Euler (x=1.5439, y=-0.0000, z=2.5874), order='XYZ'>).png":
        return "leftfront"
    
class Remote60(IterableDataset):
    def __init__(self, root_dir, descriptions_file=None, is_test=False, is_val=False, is_train=False):
        self.is_test = is_test
        self.is_val = is_val
        self.is_train = is_train
        self.root_dir = root_dir
        self.desired_size = (800, 800)

        self.descriptions_file = descriptions_file
        if self.descriptions_file is not None:
            self.descriptions = self.load_descriptions(self.descriptions_file)

        self.labels = []
        
        self.class_to_coding = {
            "topleftfront": [0, 0, 0],
            "topleftback": [0, 0, 1],
            "topleft": [0, 0, 2],
            "toprightfront": [0, 1, 0],
            "toprightback": [0, 1, 1],
            "topright": [0, 1, 2],
            "topfront": [0, 2, 0],
            "topback": [0, 2, 1],
            "top": [0, 2, 2],
            "bottomleftfront": [1, 0, 0],
            "bottomleftback": [1, 0, 1],
            "bottomleft": [1, 0, 2],
            "bottomrightfront": [1, 1, 0],
            "bottomrightback": [1, 1, 1],
            "bottomright": [1, 1, 2],
            "bottomfront": [1, 2, 0],
            "bottomback": [1, 2, 1],            
            "bottom": [1, 2, 2],
            "leftfront": [2, 0, 0],
            "leftback": [2, 0, 1],
            "left": [2, 0, 2],
            "rightfront": [2, 1, 0],
            "rightback": [2, 1, 1],
            "right": [2, 1, 2],
            "front": [2, 2, 0],
            "back": [2, 2, 1],
            "null": [2, 2, 2]
        }

        self.classes = list(self.class_to_coding.keys())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.transform_train = transforms.Compose([
            transforms.Resize(self.desired_size),
            transforms.ToTensor(),  
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomResizedCrop(size=self.desired_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
            ], p=0.2),
        ])

        self.transform = transforms.Compose([
            transforms.Resize(self.desired_size),
            transforms.ToTensor()
        ])
        
    def load_descriptions(self, descriptions_file):
        with open(descriptions_file, "r") as f:
            descriptions = json.load(f)
        return descriptions
    
    def __iter__(self):
        path = self.get_data_path()
        for subdir, _, files in os.walk(path):
            for file in files:
                if file.endswith('.png'):
                    img_path = os.path.join(subdir, file)
                    label = assign_label(file)

                    image = self.load_image(img_path)
                    label_coding = self.class_to_idx[label]

                    if self.descriptions_file is not None:
                        description = self.descriptions[img_path]
                        yield image, label_coding, description
                    else:
                        yield image, label_coding

    def get_data_path(self):
        if self.is_val:
            return os.path.join(self.root_dir, 'val')
        elif self.is_test:
            return os.path.join(self.root_dir, 'test')
        elif self.is_train:
            return os.path.join(self.root_dir, 'train')
        else:
            raise ValueError("Error: Invalid dataset split")

    def load_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        if self.is_train:
            image = self.transform_train(image)
        else:
            image = self.transform(image)
        return image
    
    def get_class_labels(self):
        return self.classes
    
    def get_all_image_paths(self):
        return self.image_paths
    
class Remote60_seq(Dataset):
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

        self.transform_train = transforms.Compose([
            transforms.Resize(self.desired_size),
            transforms.ToTensor(),  
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomResizedCrop(size=self.desired_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
            ], p=0.2),
        ])

        self.transform = transforms.Compose([
            transforms.Resize(self.desired_size),
            transforms.ToTensor()
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
                    label = assign_label(file)

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
        if self.is_train:
            image = self.transform_train(image)
        else:
            image = self.transform(image)
        return image

class Remote14(IterableDataset):
    def __init__(self, root_dir, descriptions_file=None, is_test=False, is_val=False, is_train=False):
        self.is_test = is_test
        self.is_val = is_val
        self.is_train = is_train
        self.root_dir = root_dir
        self.desired_size = (800, 800)

        self.descriptions_file = descriptions_file
        if self.descriptions_file is not None:
            self.descriptions = self.load_descriptions(self.descriptions_file)

        self.labels = []
        self.classes = ['back', 'bottom', 'bottomleftback', 'bottomleftfront', 
                        'bottomrightback', 'bottomrightfront', 'front', 'left', 
                        'right', 'top', 'topleftback', 'topleftfront', 'toprightback', 'toprightfront']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.transform_train = transforms.Compose([
            transforms.Resize(self.desired_size),
            transforms.ToTensor(),  
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomResizedCrop(size=self.desired_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
            ], p=0.2),
        ])

        self.transform = transforms.Compose([
            transforms.Resize(self.desired_size),
            transforms.ToTensor()
        ])

    def load_descriptions(self, descriptions_file):
        with open(descriptions_file, "r") as f:
            descriptions = json.load(f)
        return descriptions
    
    def __iter__(self):
        path = self.get_data_path()
        for subdir, _, files in os.walk(path):
            for file in files:
                if file.endswith('.png'):
                    img_path = os.path.join(subdir, file)
                    label = os.path.splitext(file)[0].lower()

                    if '_rot' in label:
                        label = label.split('_rot')[0]

                    image = self.load_image(img_path)
                    label_idx = self.class_to_idx[label]

                    if self.descriptions_file is not None:
                        description = self.descriptions[img_path]
                        yield image, label_idx, description
                    else:
                        yield image, label_idx

    def get_data_path(self):
        if self.is_val:
            return os.path.join(self.root_dir, 'val')
        elif self.is_test:
            return os.path.join(self.root_dir, 'test')
        elif self.is_train:
            return os.path.join(self.root_dir, 'train')
        else:
            raise ValueError("Error: Invalid dataset split")

    def load_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        if self.is_train:
            image = self.transform_train(image)
        else:
            image = self.transform(image)
        return image
    
    def get_class_labels(self):
        return self.classes
    
class Remote14_seq(Dataset):
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

        self.transform_train = transforms.Compose([
            transforms.Resize(self.desired_size),
            transforms.ToTensor(),  
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            ], p=0.5),
            transforms.RandomApply([
                transforms.RandomResizedCrop(size=self.desired_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
            ], p=0.2),
        ])

        self.transform = transforms.Compose([
            transforms.Resize(self.desired_size),
            transforms.ToTensor()
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
        if self.is_train:
            image = self.transform_train(image)
        else:
            image = self.transform(image)
        return image
    