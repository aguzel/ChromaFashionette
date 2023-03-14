import torch 
import torchvision.transforms as transforms
from preprocessing import FashionImageSegmentationDataset
import torch.nn as nn
from models.unet import UnetGenerator
from models.fcn import *
from models.deeplabv3 import deeplabV3
from models.GSCNN.naive import GSCNN
from utils import *
from tqdm import tqdm

import torch.utils.tensorboard as tb
writer = tb.SummaryWriter('runs/')

# Training Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NORMALIZE = False
ARCHITECTURE = 'GSCNN'
NUM_CLASSES = 5 
LR = 1e-4
EPOCHS = 10

# Data Load
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

# Network
if ARCHITECTURE == 'U-net':
  net = UnetGenerator(3, 5, 7, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False)
elif ARCHITECTURE == 'FCN':
  vgg_model = VGGNet(requires_grad=True, remove_fc=True)
  net = FCNs(pretrained_net=vgg_model, n_class=NUM_CLASSES)
elif ARCHITECTURE == 'DeeplabV3+':
  net = deeplabV3(n_class = NUM_CLASSES)
elif ARCHITECTURE == 'GSCNN':
  net = GSCNN(n_class = NUM_CLASSES)

net = net.to(device)

# Optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),
                             lr=LR,
                             betas=(0.9, 0.999),
                             eps=1e-08,
                             weight_decay=0,
                             )

# Training
net.train()
for epoch in range(EPOCHS):
    training_loss = 0.0
    t = tqdm(enumerate(trainloader))
    for i, data in t:
        description = 'TRAIN : {:.1f}%'.format((i + 1) / len(trainloader) * 100)
        t.set_description(description)
        inputs, targets, _ = data
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)        
        predictions = net(inputs)        
        loss = loss_func(predictions['out'], targets)        
        loss.backward(retain_graph=True)
        optimizer.step()
        training_loss += loss.item()
        if i % 160 == 159:    
          writer.add_scalar('training loss',
                          training_loss / 160,
                          epoch * len(trainloader) + i) 
          

    print('[Epoch %d/%d] Training Loss: %.4f' % (epoch + 1, EPOCHS, training_loss / len(trainloader)))

# Save the network
PATH = "StateDictionary/trained_{}_LR_:{}_EPOCH_:{}.pt".format(ARCHITECTURE, LR, EPOCHS)
torch.save(net.state_dict(), PATH )
print("The End of Training and model saved to {}".format(PATH))

