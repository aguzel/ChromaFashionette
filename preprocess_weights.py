import torch 
import torchvision.transforms as transforms
from preprocessing import FashionImageSegmentationDataset
from utils import calculate_class_weights

transform = transforms.Compose([
        transforms.ToTensor()
    ])

trainset = FashionImageSegmentationDataset(root_dir='data', mode='train', transform=transform, normalize=False)
trainloader_for_weights = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True, num_workers=4)

for data in trainloader_for_weights:
        inputs, targets, gt_image = data
print(gt_image.shape)

weights = calculate_class_weights(gt_image)
print(weights)