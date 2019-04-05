import argparse
from PIL import Image

from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch

from gradcam import GradCam

import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Testing Gradient Class Activation Map')
parser.add_argument('--image', default='both.png', type=str)
parser.add_argument('--opimage', default='bothcam.png', type = str)
parser.add_argument('--contrib', default=0, type=int)
parser.add_argument('--opheatmap', default='bothcam_heatmap.png', type = str)
# parser.add_argument('--opmask', default='grad_cam_dog.png', type = str)


args = parser.parse_args()


img_name = args.image
op_name = args.opimage
index = args.contrib
ophm_name = args.opheatmap
# opm_name = args.opmask

img = Image.open(img_name)

# Preprocessing applied on image
preprocess = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
)
img = preprocess(img).unsqueeze(0)

# Model used is VGG 19
model = models.vgg19(pretrained=True)


# Pass to available device
if torch.cuda.is_available():
    img = img.cuda()
    model.cuda()
else:
    img = img.cpu()
    model.cpu()

# __init__ function called now
grad_cam = GradCam(model)
# __call__ function called now
feature_img = grad_cam(img, index)
feature_img = transforms.ToPILImage()(feature_img[0])


feature_img.save(op_name)
