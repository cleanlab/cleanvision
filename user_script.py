#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:06:23 2022

@author: yimingchen
"""
import time 
from utils import *
from ImageDataset import ImageDataset


start = time.time()
imgset = ImageDataset((128,128))

print("The path is:", imgset.path) #Question: How to stop this from triggering the progress bar?
print("There are ", str(imgset.image_num), "images in the dataset")
print(imgset.audit_images(True)) #returns the dictionary of issues and pandas dataframe of information


end = time.time()
total_time = end - start
print("\n"+ str(total_time))

#TODO: ask about creating an examples folder and putting user_script there?

