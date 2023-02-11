# %%
import torch 
import torchvision.transforms as transforms
from preprocessing import FashionImageSegmentationDataset
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from unet import UnetGenerator
from fcn import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_image_dir = "/data"
transform = transforms.Compose([transforms.ToTensor()])
dataset = FashionImageSegmentationDataset(root_dir='data', mode='train', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)


# %%
# Display the first batch of images
images, labels, gt_images = next(iter(dataloader))
grid = torchvision.utils.make_grid(images, nrow=8)
plt.figure(figsize=(15, 15))
plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.show()

grid = torchvision.utils.make_grid(gt_images, nrow=8)
plt.figure(figsize=(15, 15))
plt.imshow(grid.numpy().transpose((1, 2, 0)))
plt.axis('off')
plt.show()

gt_label = labels.unsqueeze(1)
grid = torchvision.utils.make_grid(gt_label, nrow=8)
plt.figure(figsize=(15, 15))
print(grid.shape)
plt.imshow(grid[0], cmap='gray')
plt.axis('off')
plt.show()

# %% create network
# net = UnetGenerator(3, 5, 7, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
# net = net.to(device)
# print(net)

vgg_model = VGGNet(requires_grad=True, remove_fc=True)
net = FCNs(pretrained_net=vgg_model, n_class=5)
net = net.to(device)
print(net)
# %%
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),
                             lr=0.0002,
                             betas=(0.9, 0.999),
                             eps=1e-08,
                             weight_decay=0,
                             )
                             

# %%
epoch = 5
net.train()
for epoch in range(epoch):
    running_loss = 0.0
    for inputs, targets, _ in dataloader:
        optimizer.zero_grad()

        inputs = images.to(device)
        targets = labels.to(device)        
        predictions = net(inputs)        
        loss = loss_func(predictions, targets)
        
        loss.backward(retain_graph=True)
        optimizer.step()

        running_loss += loss.item()
    print('[Epoch %d/%d] Loss: %.4f' % (epoch + 1, epoch, running_loss / len(dataloader)))
torch.save(net.state_dict(), "fcn_vgg_model.pt")

# %%
plt.imshow(images[7].detach().cpu().numpy().transpose(1,2,0))
# %%
plt.imshow(gt_images[7].detach().cpu().numpy().transpose(1,2,0))
# %%
# plt.imshow(labels[7], cmap='gray')
example_result  = (net(images[7].unsqueeze(0).to(device)))
out_1 = torch.argmax(example_result.squeeze(), dim=0).detach().cpu().numpy() 

# %%
print(out_1.shape)
plt.imshow(out_1)
plt.savefig('out.png')
# %%
# Define the helper function
def decode_segmap(image, nc=5):
  label_colors = np.array([(0, 0, 0),
                           (66, 127, 50),
                           (224, 16, 115),
                           (250, 190, 8),
                           (147, 225, 245)])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
  rgb = np.stack([r, g, b], axis=2)
  return rgb

rgb = decode_segmap(out_1)
plt.imshow(rgb)