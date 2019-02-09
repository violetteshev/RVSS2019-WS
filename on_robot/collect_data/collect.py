#!/usr/bin/env python3
import time
import click
import math
import sys
sys.path.append("..")
import cv2
import numpy as np
import penguinPi as ppi
import pygame
import torch
from torch import nn


#~~~~~~~~~~~~ SET UP Game ~~~~~~~~~~~~~~
pygame.init()
pygame.display.set_mode((100, 100))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# stop the robot 
ppi.set_velocity(0, 0)

try:
    lead = 0
    Kd = 16 # general  speed
    Ka = 2 # how much speed is changed when turning
    left = Kd # left wheel speed
    right = Kd # right wheel speed
    while True:

        # Get an image from the the robot.
        image = ppi.get_image()
        
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    print("straight")
                    left = Kd
                    right = Kd
                if event.key == pygame.K_DOWN:
                    left = Kd
                    right = Kd
                if event.key == pygame.K_RIGHT:
                    print("right")
                    if left > right:
                        left += Ka
                        right -= Ka
                    else:
                        left = Kd + Ka
                        right = Kd - Ka
                if event.key == pygame.K_LEFT:
                    print("left")
                    if right > left:
                        right += Ka
                        left -= Ka
                    else:
                        right = Kd + Ka
                        left = Kd - Ka
                if event.key == pygame.K_SPACE:
                    print("stop")                    
                    ppi.set_velocity(0,0)
                    raise KeyboardInterrupt
        
        ppi.set_velocity(left, right) 

        cv2.imwrite("data/"+str(lead).zfill(6)+'_{}_{}.jpg'
                                                .format(left, right), image) 
        lead += 1
        
        
except KeyboardInterrupt:    
    ppi.set_velocity(0, 0)
