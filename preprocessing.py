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
            gray_scale = torch.mean(gt_image, dim=0)
            gray_scale = gray_scale.view(512*256)
            min_values = torch.min(gray_scale, dim=0, keepdim=True).values
            max_values = torch.max(gray_scale, dim=0, keepdim=True).values
            gray_scale = (gray_scale - min_values) / (max_values - min_values)
            thresholds = torch.tensor([0.,
                                    0.39384118,
                                    0.57536465,
                                    0.726094,
                                    1.])
            class_labels = torch.zeros_like(gray_scale)
            for i in range(5):
                class_labels[(gray_scale == thresholds[i])] = i
            class_labels = class_labels.view(512, 256)
            class_labels = class_labels.long()

        return input_image, class_labels, gt_image





        