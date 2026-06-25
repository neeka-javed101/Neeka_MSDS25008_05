
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

# animal dataset class to load images from the specified directory structure
class AnimalDataset(Dataset):

    def __init__(self, root_dir,max_images_per_class=20,num_classes=5):
    

        self.root_dir = root_dir
        self.images = []
        self.max_images_per_class = max_images_per_class
        self.num_classes = num_classes
# Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])
        self.valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG']
        class_count = 0
        img_count_per_class = 0

# Load images from the specified directory structure
        for label in os.listdir(root_dir):
            if class_count >= self.num_classes:
                break

            label_dir = os.path.join( root_dir, label)

            if os.path.isdir(label_dir):
                img_count_per_class = 0
# Load images for the current class
                for img_name in os.listdir(label_dir):
                    if os.path.splitext(img_name)[1].lower() not in self.valid_extensions:
                        continue
                    img_path = os.path.join(label_dir, img_name)
                    try:
                        img = Image.open(img_path) 
                        img.verify()  # Verify that the image is not corrupted
                        
                        self.images.append(img_path)
                        img_count_per_class += 1
                          # Verify that the image is not corrupted

                        if img_count_per_class >= self.max_images_per_class:
                          break
                    except Exception as e:
                     print(f"Error loading image {img_name}")
                     continue
                if img_count_per_class > 0:
                    class_count += 1
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert("RGB")
            image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image: {self.images[idx]}")
            return self.__getitem__((idx + 1) % len(self.images))