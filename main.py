# %%
import torch 
import torchvision.transforms as transforms
from preprocessing import FashionImageSegmentationDataset
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from models.unet import UnetGenerator
from models.fcn import *
from utils import *
import torch.utils.tensorboard as tb
writer = tb.SummaryWriter('runs/fcn32s-v2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

BATCH_SIZE = 8
NORMALIZE = False


transform_norm = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

transform = transforms.Compose([
        transforms.ToTensor()
    ])

if NORMALIZE:
  transform_data = transform_norm
else:
  transform_data = transform


trainset = FashionImageSegmentationDataset(root_dir='data', mode='train', transform=transform_data, normalize=NORMALIZE)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

testset = FashionImageSegmentationDataset(root_dir='data', mode='test', transform=transform_data, normalize=NORMALIZE)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# %%
# Display the first batch of images
torch.set_printoptions(precision=7)
images, labels, gt_images = next(iter(trainloader))
show_grid_images(images, nrow=8)
show_grid_images(gt_images, nrow=8)
show_grid_labels(labels, nrow=8)

# %% create network
# net = UnetGenerator(3, 5, 7, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
# net = net.to(device)
# print(net)
#%%
vgg_model = VGGNet(requires_grad=True, remove_fc=True)
net = FCNs(pretrained_net=vgg_model, n_class=5)
net = net.to(device)


loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),
                             lr=0.0002,
                             betas=(0.9, 0.999),
                             eps=1e-08,
                             weight_decay=0,
                             )                         

# %%
epochs = 9
net.train()
for epoch in range(epochs):
    training_loss = 0.0
    test_loss = 0.0
    pixel_acc = 0.0
    iou = 0.0
    for i, data in enumerate(trainloader):
        inputs, targets, _ = data
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)        
        predictions = net(inputs)        
        loss = loss_func(predictions, targets)        
        loss.backward(retain_graph=True)
        optimizer.step()
        training_loss += loss.item()
        if i % 160 == 159:    
          writer.add_scalar('training loss',
                          training_loss / 160,
                          epoch * len(trainloader) + i) 

    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, targets, gt_rgb = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = net(inputs)
            loss = loss_func(predictions, targets)
            test_loss += loss.item()
            preds_ = torch.argmax(predictions.squeeze(), dim=1)
            pixel_acc += pixel_accuracy(decode_output(preds_).detach().cpu(), gt_rgb, background_count=False)
            iou += intersection_over_unit((preds_).detach().cpu(), targets.detach().cpu())
            if i % 40 == 39:    
              writer.add_scalar('test loss',
                              test_loss / 40,
                              epoch * len(testloader) + i) 
            

        print('[Epoch %d/%d] Training Loss: %.4f Test Loss: %.4f Pixel Acc: %.3f IOU: %.3f' % (epoch + 1, epochs,
                                                                     training_loss / len(trainloader),
                                                                     test_loss / len(testloader),
                                                                     pixel_acc / len(testloader),
                                                                     iou / len(testloader)
                                                                     ))
PATH = "trained_fcn32s.pt"
torch.save(net.state_dict(), PATH )
print("The End of Training and model saved to {}".format(PATH))
#%%
net.load_state_dict(torch.load("trained_fcn32s.pt"))
net.eval()
#%%
testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True, num_workers=4)
images, labels, gt_images = next(iter(testloader))
show_grid_images(images.detach().cpu())
show_grid_images(gt_images.detach().cpu()[1:2])
example_results = (net(images.to(device)))
preds_ = torch.argmax(example_results.squeeze(), dim=1)
output_images = decode_output(preds_)
show_grid_images(output_images.detach().cpu()[1:2])
print(pixel_accuracy(decode_output(preds_).detach().cpu()[1:2],gt_images[1:2], background=False ))
print(pixel_accuracy(decode_output(preds_).detach().cpu()[1:2],gt_images[1:2], background=True ))
# print(iou((preds_).detach().cpu(), labels ))

# %%
# %%
print(pixel_accuracy(decode_output(preds_).detach().cpu(),gt_images ))
print(pixel_accuracy(gt_images, gt_images ))
print(intersection_over_unit(labels.detach().cpu(), labels.detach().cpu()))
# %%
import odak
from utils import *
gt = odak.learn.tools.load_image('/home/atlas/ChromaFashionette/gt.png', normalizeby = 255.)
test = odak.learn.tools.load_image('/home/atlas/ChromaFashionette/test_025x.png', normalizeby = 255.)
gt = gt.swapaxes(0,2)
test = test.swapaxes(0,2)
gt_stack = torch.stack((gt, gt, gt, gt, gt), dim=0)
test_stack = torch.stack((test, test, test, test, test), dim=0)
print(pixel_accuracy(test.unsqueeze(0), gt.unsqueeze(0), False))
print(pixel_accuracy(test.unsqueeze(0), gt.unsqueeze(0), True))
print(pixel_accuracy(test_stack, gt_stack, True))

# %%
