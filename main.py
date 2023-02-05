# %%
import torch 
import torchvision.transforms as transforms
from preprocessing import FashionImageSegmentationDataset
import torchvision
import matplotlib.pyplot as plt
import numpy as np

train_image_dir = "/data"
transform = transforms.Compose([transforms.ToTensor()])
dataset = FashionImageSegmentationDataset(root_dir='data', mode='train', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)



# %%
# Display the first batch of images
images, gt = next(iter(dataloader))
grid = torchvision.utils.make_grid(images, nrow=4)
plt.figure(figsize=(15, 15))
plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.show()

gt_label = gt.unsqueeze(1)
grid = torchvision.utils.make_grid(gt_label, nrow=4)
plt.figure(figsize=(15, 15))
print(grid.shape)
plt.imshow(grid[0], cmap='gray')
plt.axis('off')
plt.show()

# %%
print(images.shape)
print(gt.shape)


# %%
