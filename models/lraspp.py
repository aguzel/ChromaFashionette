def LRASPP(n_class=5):

    from torchvision.models.segmentation.deeplabv3 import DeepLabHead
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    lrassp = lraspp_mobilenet_v3_large(num_classes= n_class)
    return lrassp