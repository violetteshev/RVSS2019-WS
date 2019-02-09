#!/usr/bin/env python3
import math
import cv2
import numpy as np
import penguinPi as ppi
import torch
from torch import nn
from steerNet import SteerNet
from torchvision import transforms

def load_net(wegihts_filename):
    model = SteerNet(5)
    model.load_state_dict(torch.load(wegihts_filename))
    model.eval()        
    return model

def preprocess_img(img,transform):
    img = transform(img)
    # add dimension for batch
    img = img.unsqueeze(0)
    return img


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
net = load_net("steerNet.pt")


#~~~~~~~~~~~~ SET UP ROBOT ~~~~~~~~~~~~~~
ppi.set_velocity(0,0)

try:
    Kd = 16
    Ka = 2
    left = Kd
    right = Kd
    while True:
        # Read image from camera.
        image = ppi.get_image()
        
        # Crop and resize to fit network.
        image = image[80:, :]
        image = cv2.resize(image, (84,84))
        # Preprocess image.
        input_img = preprocess_img(image,transform)
        # Get turn prediction.
        res = net(input_img).data.numpy()
        turn = np.argmax(res)        
        print(turn)

        # Find wheels speed values.
        left  = int(Kd - Ka*2 + turn*Ka)
        right = int(Kd + Ka*2 - turn*Ka)
        
        ppi.set_velocity(left, right)
        
        
except KeyboardInterrupt:
    ppi.set_velocity(0,0)
