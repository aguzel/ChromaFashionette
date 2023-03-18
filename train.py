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
import time

import torch.utils.tensorboard as tb
writer = tb.SummaryWriter('runs/')

# Training Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NORMALIZE = False
ARCHITECTURE = 'FCNs'
NUM_CLASSES = 5 
LR = 1e-4
EPOCHS = 20

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

valset = FashionImageSegmentationDataset(root_dir='data', mode='test', transform=transform_data, normalize=NORMALIZE)
valoader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Network
if ARCHITECTURE == 'U-Net_357_192':
  net = UnetGenerator(3, 5, 7, ngf=192, norm_layer=nn.BatchNorm2d, use_dropout=False)
elif ARCHITECTURE == 'FCNs':
  vgg_model = VGGNet(requires_grad=True, remove_fc=True)
  net = FCNs(pretrained_net=vgg_model, n_class=NUM_CLASSES)
  # net = FCN32s(pretrained_net=vgg_model, n_class=NUM_CLASSES)
  # net = FCN16s(pretrained_net=vgg_model, n_class=NUM_CLASSES)
  # net = FCN8s(pretrained_net=vgg_model, n_class=NUM_CLASSES)
elif ARCHITECTURE == 'DeeplabV3+':
  net = deeplabV3(n_class = NUM_CLASSES)
elif ARCHITECTURE == 'GSCNN':
  net = GSCNN(num_classes= NUM_CLASSES)

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
scaler = torch.cuda.amp.GradScaler()
start_time = time.time()
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
        with torch.autocast(device_type='cuda', dtype=torch.float16):
           predictions = net(inputs)
        # if ARCHITECTURE == 'DeeplabV3+':
        #   loss = loss_func(predictions['out'], targets)
        # else:
           loss = loss_func(predictions, targets)   
        # loss.backward(retain_graph=True)
        scaler.scale(loss).backward()
        # optimizer.step()
        scaler.step(optimizer)
        scaler.update()
        training_loss += loss.item()
        if i % 200 == 199:    
          writer.add_scalar('training loss',
                          training_loss / 199,
                          epoch * len(trainloader) + i) 
          

    print('[Epoch %d/%d] Training Loss: %.4f' % (epoch + 1, EPOCHS, training_loss / len(trainloader)))
    test_loss = 0.0
    pixel_acc = 0.0
    iou = 0.0
    net.eval()
    t = tqdm(enumerate(valoader))
    for i, data in t:
        description = 'VAL : {:.1f}%'.format((i + 1) / len(valoader) * 100)
        t.set_description(description)
        inputs, targets, gt_rgb = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = net(inputs)
        if ARCHITECTURE == 'DeeplabV3+':
          loss = loss_func(predictions['out'], targets)
          preds_ = torch.argmax(predictions['out'].squeeze(), dim=1)
        else:
          loss = loss_func(predictions, targets)
          preds_ = torch.argmax(predictions.squeeze(), dim=1)   
        test_loss += loss.item()             
        pixel_acc += pixel_accuracy(decode_output(preds_).detach().cpu(), gt_rgb, background_count=False)
        iou += intersection_over_unit((preds_).detach().cpu(), targets.detach().cpu())

    print('Val. Loss: %.4f Pixel Acc: %.3f IOU: %.3f' % ( test_loss / len(valoader),
                                                          pixel_acc / len(valoader),
                                                          iou / len(valoader)
                                                          ))
    print('-------------------------------\n', '-------------------------------')

end_time = time.time()
print(f"Running time: {(end_time - start_time):.5f} seconds")

# Save the network
PATH = "StateDictionary/trained_{}_LR_:{}_EPOCH_:{}.pt".format(ARCHITECTURE, LR, EPOCHS)
torch.save(net.state_dict(), PATH )
print("The End of Training and model saved to {}".format(PATH))

