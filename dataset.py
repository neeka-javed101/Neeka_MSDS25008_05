import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AnimalDataset(Dataset):

    def __init__(self, root_dir):

        self.root_dir = root_dir
        self.images = []

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

        for label in os.listdir(root_dir):

            label_dir = os.path.join(
                root_dir,
                label
            )

            if os.path.isdir(label_dir):

                for img_name in os.listdir(label_dir):

                    self.images.append(
                        os.path.join(
                            label_dir,
                            img_name
                        )
                    )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(
            self.images[idx]
        ).convert("RGB")

        image = self.transform(image)

        return image