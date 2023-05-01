import torch 
import torchvision.transforms as transforms
from preprocessing import FashionImageSegmentationDataset
import torch.nn as nn
from models.unet import UnetGenerator
from models.fcn import *
from models.deeplabv3 import deeplabV3
from models.GSCNN.gscnn import GSCNN
from models.lraspp import LRASPP
from utils import *
from tqdm import tqdm


import torch.utils.tensorboard as tb
writer = tb.SummaryWriter('runs/')


# Training Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NORMALIZE = False
ARCHITECTURE = 'DeeplabV3+'
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
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

if ARCHITECTURE == 'U-Net':
  net = UnetGenerator(3, 5, 7, ngf=128, norm_layer=nn.BatchNorm2d, use_dropout=False)
elif ARCHITECTURE == 'FCNs':
  vgg_model = VGGNet(requires_grad=True, remove_fc=True)
  net = FCNs(pretrained_net=vgg_model, n_class=NUM_CLASSES)
  # net = FCN32s(pretrained_net=vgg_model, n_class=NUM_CLASSES)
  # net = FCN16s(pretrained_net=vgg_model, n_class=NUM_CLASSES)
  # net = FCN8s(pretrained_net=vgg_model, n_class=NUM_CLASSES)
elif ARCHITECTURE == 'DeeplabV3+':
  net = deeplabV3(n_class = NUM_CLASSES)
elif ARCHITECTURE =='GSCNN':
   net = GSCNN(num_classes=NUM_CLASSES)
elif ARCHITECTURE == 'LRASPP':
  net = LRASPP(n_class = NUM_CLASSES)

net = net.to(device)
net.load_state_dict(torch.load("StateDictionary/trained_DeeplabV3+_LR_:0.0001_EPOCH_:15_weighted.pt"))
net.eval()

loss_func = nn.CrossEntropyLoss()

test_loss = 0.0
pixel_acc = 0.0
pixel_acc_wo_bg = 0.0
pixel_acc_class = np.zeros(5)
iou = 0.0
t = tqdm(enumerate(testloader))
for i, data in t:
    description = 'TEST : {:.1f}%'.format((i + 1) / len(testloader) * 100)
    t.set_description(description)
    inputs, targets, gt_rgb = data
    inputs = inputs.to(device)
    targets = targets.to(device)
    predictions = net(inputs)
    if ARCHITECTURE == 'DeeplabV3+' or ARCHITECTURE == 'LRASPP':
       predictions = predictions['out']
    loss = loss_func(predictions, targets)
    test_loss += loss.item()
    preds_ = torch.argmax(predictions.squeeze(), dim=1)
    pixel_acc += pixel_accuracy(decode_output(preds_).detach().cpu(), gt_rgb, background_count=True)
    pixel_acc_wo_bg += pixel_accuracy(decode_output(preds_).detach().cpu(), gt_rgb, background_count=False)
    pixel_acc_class += np.array(class_pixel_accuracy(decode_output(preds_).detach().cpu(), gt_rgb))
    iou += intersection_over_unit((preds_).detach().cpu(), targets.detach().cpu())
    # if i == 2:
    #   import sys; sys.exit()
    if i % 40 == 39:    
      writer.add_scalar('test loss',
                      test_loss / 40,
                      len(testloader) + i) 
print('Test Loss: %.4f Pixel Acc: %.3f %.3f IOU: %.3f' % ( test_loss / len(testloader),
                                                      pixel_acc / len(testloader),
                                                      pixel_acc_wo_bg / len(testloader),
                                                      iou / len(testloader)
                                                      ))
pixel_acc_class = pixel_acc_class / len(testloader)
print('Background Accuracy: %.3f\n Hair Accuracy: %.3f\n Clothes Accuracy: %.3f\n Skin Accuracy: %.3f\n Accessories Accuracy: %.3f\n' % (pixel_acc_class[0],
                                                                                                                                         pixel_acc_class[1],
                                                                                                                                         pixel_acc_class[2],
                                                                                                                                         pixel_acc_class[3],
                                                                                                                                         pixel_acc_class[4]
                                                                                                                                         ))


print('Image Report is being generated...')
# Image Report
images, labels, gt_images = next(iter(testloader))
predictions = (net(images.to(device)))
if ARCHITECTURE == 'DeeplabV3+' or ARCHITECTURE == 'LRASPP':
    predictions = predictions['out']
preds_ = torch.argmax(predictions.squeeze(), dim=1)
output_images = decode_output(preds_)
print("Writing to report folder...")
show_grid_images(images.detach().cpu(),nrow=BATCH_SIZE, save='report/_input.png', legend='INPUT')
show_grid_labels(labels.detach().cpu(), nrow=BATCH_SIZE, save='report/_labels.png')
show_grid_images(gt_images.detach().cpu(), nrow=BATCH_SIZE, save='report/ground_truth.png', legend='GROUND TRUTH')
show_grid_images(output_images.detach().cpu(), nrow=BATCH_SIZE, save='report/predictions.png', legend='Predictions')