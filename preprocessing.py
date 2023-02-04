import os
import torch
import torchvision.transforms as transforms
from PIL import Image

class FashionImageSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        self.input_image_paths = []
        self.gt_image_paths = []
        if self.mode == 'train':
            input_data_dir = os.path.join(root_dir, 'train/A')
            gt_data_dir = os.path.join(root_dir, 'train/B')
        elif self.mode == 'test':
            input_data_dir = os.path.join(root_dir, 'test/A')
            gt_data_dir = os.path.join(root_dir, 'test/B')

        for filename in os.listdir(input_data_dir):
            if filename.endswith(".png"):
                self.input_image_paths.append(os.path.join(input_data_dir, filename))
                self.gt_image_paths.append(os.path.join(gt_data_dir, filename))

    def __len__(self):
        return len(self.input_image_paths)

    def __getitem__(self, idx):
        input_image = Image.open(self.input_image_paths[idx])        
        gt_image = Image.open(self.gt_image_paths[idx])

        if self.transform:
            input_image = self.transform(input_image)
            gt_image = self.transform(gt_image)

        return input_image, gt_image