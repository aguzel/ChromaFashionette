import torchvision
from matplotlib import pyplot as plt
import numpy as np
import torch

def show_grid_images(images, nrow = 8, save = 'image.png', legend = 'ground_truth'):
    grid = torchvision.utils.make_grid(images, nrow)
    plt.figure(figsize=(15, 15), facecolor='black')
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title(legend)
    plt.axis('off')
    plt.savefig(save)
    return None


def show_grid_labels(labels, nrow = 8, save = 'image.png'):
    gt_label = labels.unsqueeze(1)
    grid = torchvision.utils.make_grid(gt_label, nrow)
    plt.figure(figsize=(15, 15), facecolor='black')
    plt.imshow(grid[0], cmap='gray')
    plt.title('LABELS')
    plt.axis('off')
    plt.savefig(save)
    return None

def decode_output(images, nc=5, device = 'cuda'):
  label_colors = torch.tensor([(0, 0, 0),
                           (66, 127, 50),
                           (224, 16, 115),
                           (250, 190, 8),
                           (147, 225, 245)])
  image_list =[]
  rgb_batch = torch.zeros(images.shape[0], 3, images.shape[1], images.shape[2])
  for i in range(images.shape[0]):
    r = torch.zeros_like(images[i]).to(torch.uint8)
    g = torch.zeros_like(images[i]).to(torch.uint8)
    b = torch.zeros_like(images[i]).to(torch.uint8)
    for l in range(0, nc):
        idx = images[i] == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = torch.stack([r, g, b], dim=0)
    image_list.append(rgb)
    rgb_batch = torch.stack(image_list)
  return rgb_batch


def pixel_accuracy(pred, target, background_count = False):
    """
    Calculates the pixel accuracy between two RGB images.
    
    Arguments:
        pred: PyTorch tensor, shape (N, 3, H, W), the predicted image.
        target: PyTorch tensor, shape (N, 3, H, W), the ground truth image.
        
    Returns:
        accuracy: float, the pixel accuracy.
    """
    # Convert the RGB images to grayscale
    target = target * 255.0
    pred = 0.2989 * pred[:, 0, :, :] + 0.5870 * pred[:, 1, :, :] + 0.1140 * pred[:, 2, :, :]
    target = 0.2989 * target[:, 0, :, :] + 0.5870 * target[:, 1, :, :] + 0.1140 * target[:, 2, :, :]
    
    pred = torch.round(pred, decimals=1)
    target = torch.round(target, decimals=1)
    classes = torch.unique(target) 
    # Calculate the number of correctly classified pixels
    background_image = torch.zeros_like(target)
    correct = (pred == target).sum().item()
    background_pixels  = (pred == background_image).sum().item()
    # Calculate the total number of pixels
    if background_count == False:
       accuracy = (correct - background_pixels) /  (pred.numel() - background_pixels)
    else:
       accuracy = correct / target.numel()
    return accuracy

def class_pixel_accuracy(pred, target):
    # Convert the RGB images to grayscale
    target = target * 255.0
    pred = 0.2989 * pred[:, 0, :, :] + 0.5870 * pred[:, 1, :, :] + 0.1140 * pred[:, 2, :, :]
    target = 0.2989 * target[:, 0, :, :] + 0.5870 * target[:, 1, :, :] + 0.1140 * target[:, 2, :, :]
    pred = torch.round(pred, decimals=1)
    target = torch.round(target, decimals=1)  
    classes = torch.unique(target)
    background = round(classes[0].item(), 1)
    skin = round(classes[1].item(), 1)
    accessories = round(classes[2].item(), 1)
    clothes = round(classes[3].item(), 1)
    hair = round(classes[4].item(), 1)
    # background
    total_background = torch.count_nonzero(target == background)
    correct_background = torch.count_nonzero((pred == background) & (target == background))
    acc_background = correct_background / total_background
    # hair
    total_hair = torch.count_nonzero(target == hair)
    correct_hair = torch.count_nonzero((pred == hair) & (target == hair))
    acc_hair = correct_hair / total_hair
    # clothes
    total_clothes = torch.count_nonzero(target == clothes)
    correct_clothes = torch.count_nonzero((pred ==clothes) & (target == clothes))
    acc_clothes = correct_clothes / total_clothes
    # skin
    total_skin = torch.count_nonzero(target == skin)
    correct_skin = torch.count_nonzero((pred == skin) & (target == skin))
    acc_skin = correct_skin / total_skin
    # accessories
    total_accessories = torch.count_nonzero(target == accessories)
    correct_accessories = torch.count_nonzero((pred == accessories) & (target == accessories))
    acc_accessories = correct_accessories / total_accessories
    
    return acc_background.item(), acc_hair.item(), acc_clothes.item(), acc_skin.item(), acc_accessories.item()
   

def intersection_over_unit(pred, target, num_classes = 5):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)
  # background class "0" is ignored 
  for cls_ in range(1, num_classes):  
    pred_inds = pred == cls_
    target_inds = target == cls_
    intersection = (pred_inds[target_inds]).long().sum().data.cpu() 
    union = pred_inds.long().sum().data.cpu() + target_inds.long().sum().data.cpu() - intersection
    ious.append(float(intersection) / float(max(union, 1)))
  return np.array(ious).mean()

def calculate_class_weights(target):
    # Convert the RGB images to grayscale
    target = target * 255.0
    target = 0.2989 * target[:, 0, :, :] + 0.5870 * target[:, 1, :, :] + 0.1140 * target[:, 2, :, :]
    target = torch.round(target, decimals=1)  
    classes = torch.unique(target)
    background = round(classes[0].item(), 1)
    skin = round(classes[1].item(), 1)
    accessories = round(classes[2].item(), 1)
    clothes = round(classes[3].item(), 1)
    hair = round(classes[4].item(), 1)
    class_weights = []
    total_background = torch.count_nonzero(target == background)
    background_ratio = total_background / target.numel()
    class_weights.append(background_ratio)


    total_skin = torch.count_nonzero(target == skin)
    skin_ratio = total_skin / target.numel()

    class_weights.append(skin_ratio)
    total_accessories = torch.count_nonzero(target == accessories)
    accessories_ratio = total_accessories / target.numel()

    class_weights.append(accessories_ratio)
    total_clothes = torch.count_nonzero(target == clothes)
    clothes_ratio = total_clothes / target.numel()

    class_weights.append(clothes_ratio)
    total_hair = torch.count_nonzero(target == hair)
    hair_ratio = total_hair / target.numel()

    class_weights.append(hair_ratio)
    # print("background ratio:{} ". format(background_ratio))
    # print("skin ratio:{} ". format(skin_ratio))
    # print("accessories ratio:{} ". format(accessories_ratio))
    # print("hair ratio:{} ". format(hair_ratio))
    # print("total ratio : {}".format(background_ratio + skin_ratio + accessories_ratio + clothes_ratio + hair_ratio))
    loss_weights = (1  - torch.tensor(class_weights)) / (len(class_weights) - 1)
    return loss_weights