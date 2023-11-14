from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class KimchiDataset(Dataset):
    def __init__(self, data_dir, transform = None):
        self.data_dir = data_dir
        # Data augmentation
        self.transform = transform
        self.class_names = os.listdir(data_dir)
        self.ToTensor = transforms.ToTensor()
        self.data = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            for filename in os.listdir(class_dir):
                image_path = os.path.join(class_dir, filename)
                self.data.append((image_path, class_name))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, class_name = self.data[idx]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self.ToTensor(image)
        return image, self.class_names.index(class_name)