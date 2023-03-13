def deeplabV3(n_class=5):

    from torchvision.models.segmentation.deeplabv3 import DeepLabHead
    from torchvision import models

    deeplabv3 = models.segmentation.deeplabv3_resnet50(weights=None,
                                                    progress=True)
    deeplabv3.classifier = DeepLabHead(2048, n_class)
    return deeplabv3