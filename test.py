import torch 
import torchvision.transforms as transforms
from preprocessing import FashionImageSegmentationDataset
import torch.nn as nn
from models.unet import UnetGenerator
from models.fcn import *
from models.deeplabv3 import deeplabV3
from models.GSCNN.gscnn import GSCNN
from utils import *
from tqdm import tqdm

import torch.utils.tensorboard as tb
writer = tb.SummaryWriter('runs/')


# Training Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NORMALIZE = False
ARCHITECTURE = 'U-Net'
NUM_CLASSES = 5 

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

testset = FashionImageSegmentationDataset(root_dir='data', mode='test', transform=transform_data, normalize=NORMALIZE)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

if ARCHITECTURE == 'U-Net':
  net = UnetGenerator(3, 5, 7, ngf=192, norm_layer=nn.BatchNorm2d, use_dropout=False)
elif ARCHITECTURE == 'FCN32s':
  vgg_model = VGGNet(requires_grad=True, remove_fc=True)
  # net = FCNs(pretrained_net=vgg_model, n_class=NUM_CLASSES)
  net = FCN32s(pretrained_net=vgg_model, n_class=NUM_CLASSES)
  # net = FCN16s(pretrained_net=vgg_model, n_class=NUM_CLASSES)
  # net = FCN8s(pretrained_net=vgg_model, n_class=NUM_CLASSES)
elif ARCHITECTURE == 'DeeplabV3+':
  net = deeplabV3(n_class = NUM_CLASSES)
elif ARCHITECTURE =='GSCNN':
   net = GSCNN(num_classes=NUM_CLASSES)

net = net.to(device)
net.load_state_dict(torch.load("StateDictionary/trained_U-Net_357_192_LR_:0.0001_EPOCH_:20.pt"))
net.eval()

loss_func = nn.CrossEntropyLoss()

test_loss = 0.0
pixel_acc = 0.0
iou = 0.0
t = tqdm(enumerate(testloader))
for i, data in t:
    description = 'TEST : {:.1f}%'.format((i + 1) / len(testloader) * 100)
    t.set_description(description)
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
                      len(testloader) + i) 
print('Test Loss: %.4f Pixel Acc: %.3f IOU: %.3f' % ( test_loss / len(testloader),
                                                      pixel_acc / len(testloader),
                                                      iou / len(testloader)
                                                      ))



# Image Report
images, labels, gt_images = next(iter(testloader))
predictions = (net(images.to(device)))
preds_ = torch.argmax(predictions.squeeze(), dim=1)
output_images = decode_output(preds_)

show_grid_images(images.detach().cpu(),nrow=BATCH_SIZE, save='report/_input.png', legend='INPUT')
show_grid_labels(labels.detach().cpu(), nrow=BATCH_SIZE, save='report/_labels.png')
show_grid_images(gt_images.detach().cpu(), nrow=BATCH_SIZE, save='report/ground_truth.png', legend='GROUND TRUTH')
show_grid_images(output_images.detach().cpu(), nrow=BATCH_SIZE, save='report/predictions.png', legend='Predictions')