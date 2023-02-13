import os
import torch
import torchvision.transforms as transforms
from PIL import Image

class FashionImageSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode, transform=None, normalize = True):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.normalize = normalize

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
            gray_scale = torch.mean(gt_image * 255.0, dim=0)
            gray_scale = gray_scale.view(512*256)
            gray_scale = torch.round(gray_scale * 1000) / 1000  
            class_labels = torch.zeros_like(gray_scale)
            if self.normalize == True:
                class_labels[gray_scale == -506.4349976] = 0
                class_labels[gray_scale == -147.3029938] = 1
                class_labels[gray_scale ==  13.8000002] =  2
                class_labels[gray_scale ==  152.0559998] = 3
                class_labels[gray_scale ==  405.3229980] = 4
            else:
                class_labels[gray_scale == 0.0]         = 0
                class_labels[gray_scale == 81.0]        = 1
                class_labels[gray_scale == 118.3330002] = 2
                class_labels[gray_scale == 149.3329926] = 3
                class_labels[gray_scale == 205.6670074] = 4

            class_labels = class_labels.view(512, 256)
            class_labels = class_labels.long()

        return input_image, class_labels, gt_image





        