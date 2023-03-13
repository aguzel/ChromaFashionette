def deeplabV3():

    from torchvision.models.segmentation.deeplabv3 import DeepLabHead
    from torchvision import models

    deeplabv3 = models.segmentation.deeplabv3_resnet50(pretrained=False,
                                                    progress=True)
    deeplabv3.classifier = DeepLabHead(2048, 5)
    return deeplabv3