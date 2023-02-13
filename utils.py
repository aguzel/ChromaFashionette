import torchvision
from matplotlib import pyplot as plt
import numpy as np
import torch

def show_grid_images(images, nrow = 8):
    grid = torchvision.utils.make_grid(images, nrow)
    plt.figure(figsize=(15, 15))
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.show()
    return None

def show_grid_labels(labels, nrow = 8):
    gt_label = labels.unsqueeze(1)
    grid = torchvision.utils.make_grid(gt_label, nrow)
    plt.figure(figsize=(15, 15))
    plt.imshow(grid[0], cmap='gray')
    plt.axis('off')
    plt.show()
    return None

def decode_output(images, nc=5):
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


def pixel_accuracy(pred, target):
    """
    Calculates the pixel accuracy between two RGB images.
    
    Arguments:
        pred: PyTorch tensor, shape (N, 3, H, W), the predicted image.
        target: PyTorch tensor, shape (N, 3, H, W), the ground truth image.
        
    Returns:
        accuracy: float, the pixel accuracy.
    """
    # Convert the RGB images to grayscale
    # pred = 0.2989 * pred[:, 0, :, :] + 0.5870 * pred[:, 1, :, :] + 0.1140 * pred[:, 2, :, :]
    # target = 0.2989 * target[:, 0, :, :] + 0.5870 * target[:, 1, :, :] + 0.1140 * target[:, 2, :, :]
    
    # Convert the images to binary masks by thresholding
    # pred = (pred > 0.5).float()
    # target = (target > 0.5).float()
    
    # Calculate the number of correctly classified pixels
    correct = (pred == target).sum().item()
    
    # Calculate the total number of pixels
    total = pred.numel()
    
    accuracy = correct / total
    
    return accuracy


def iou(pred, target, n_classes = 5):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
  return np.array(ious)
