import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt


class GradCam:
    # Initialize it with the model
    def __init__(self,model):
        self.model = model
        self.feature = None
        self.gradient = None

    def save_grad(self, grad):
        self.gradient = grad

    # Object is called with an image
    def __call__(self, x, index):
        img_size = (x.size(-1), x.size(-2))
        data = Variable(x)
        heat_maps = []

        # Loads each image as numpy array first
        # Saves it for later
        # Operates on the tensor image
        for i in range(data.size(0)):
            img = data[i].data.cpu().numpy()
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img/np.max(img)

            feature = data[i].unsqueeze(0)

            # Accessing different layers
            for name, module in self.model.named_children():
                if name == 'classifier':
                    feature = feature.view(feature.size(0), -1)
                feature = module(feature)
                if name == 'features':
                    feature.register_hook(self.save_grad)
                    self.feature = feature

            # Computing class preds and backprop
            sig = nn.Sigmoid()
            classes = sig(feature)
            ind, _ = classes.topk(500)
            one_hot = ind[0][index]
            self.model.zero_grad()
            one_hot.backward()

            # print(" 1. Gradient shape : ", self.gradient.shape)
            alphas = self.gradient.mean(dim = 3, keepdim = True).mean(dim = 2, keepdim = True)
            # print(" 2. Alphas shape : ",alphas.shape)
            # print(" 3. Feature shape: ",self.feature.shape)

            mask = f.relu(alphas*self.feature).sum(dim=1).squeeze(0)

            # print(" 4. Mask shape : ", mask.shape)
            mask = cv2.resize(mask.data.cpu().numpy(), img_size)
            # print(" 5. Mask Shape, Max, Min, Mean: ", np.shape(mask), np.max(mask), np.min(mask), np.mean(mask))
            mask = mask - np.min(mask)
            # print(" 6. Mask Shape, Max, Min, Mean: ", np.shape(mask), np.max(mask), np.min(mask), np.mean(mask))
            if np.max(mask) != 0:
               mask = mask / np.max(mask)
            heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            plt.imshow(heat_map)
            plt.show()
            # print(" 7. Heat Map Shape, Max, Min : ", np.shape(heat_map), np.max(heat_map), np.min(heat_map),
            # np.mean(heat_map))
            gcam = heat_map + np.float32((np.uint8(img.transpose((1,2,0))*255)))
            gcam = gcam - np.min(gcam)
            if np.max(gcam) != 0:
                gcam = gcam/np.max(gcam)
            heat_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255*gcam), cv2.COLOR_BGR2RGB)))
        heat_maps = torch.stack(heat_maps)
        return heat_maps