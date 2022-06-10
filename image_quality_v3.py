#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 16:14:44 2022

@author: yimingchen
"""

from PIL import Image, ImageStat
import glob, math, hashlib, time, statistics
import pandas as pd
import numpy as np
from collections import OrderedDict


types = ["*.jpg", "*.jpeg", "*.gif", "*.jp2", "*.TIFF", "*.WebP","*.PNG"] #filetypes supported by PIL

class ImageDataset:
    def __init__(self, thumbnailsize):
        self.thumbnailsize = thumbnailsize
        
    def sorted_images(self):
        raw_images = []
        for type in types:
            filetype = glob.glob(type)
            if filetype == []:
                continue
            raw_images += filetype
        return sorted(raw_images) #sort image names alphabetically and numerically

        
imgset = ImageDataset((128,128))
images = imgset.sorted_images()
        

count = 0
bright = OrderedDict()
hashes = set()
dup_indices = []
image_names = []
prop = OrderedDict()
entropy = OrderedDict()

start = time.time()


for image in images:
    count+=1 #count number of images
    with open(image, 'rb') as file:
        img = Image.open(file)
        img.thumbnail(imgset.thumbnailsize) 
        
        #Analyzes image brightness
        stat = ImageStat.Stat(img)
        #r,g,b = stat.mean[0:3] 
        try:
            r,g,b = stat.mean
        except:
            print(f"WARNING: {image} does not have just r, g, b values")
        cur_bright = (math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)))/255
        bright_score = min(cur_bright, 1-cur_bright) #too bright or too dark
        bright[count-1] = bright_score
        
        #Finds duplicates
        cur_hash = hashlib.md5(img.tobytes()).hexdigest()
        if cur_hash in hashes:
            dup_indices.append(count-1) #append image index
        else:
            hashes.add(cur_hash)
            
        #Analyzes image size
        width, height = img.size
        prop_score = min(width/height, height/width) #consider extreme shapes
        prop[count-1] = prop_score
        
        #Calculates image entropy
        cur_entropy = img.entropy()/10 #rescales as 10 is the max for the entropy() function
        entropy[count-1] = cur_entropy
        
        #Obtain image name
        image_names.append(image)


    
def analyze_scores(scores): 
    mean = statistics.mean(scores.values())
    stdev = statistics.stdev(scores.values())
    sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])} #sort scores from low to high (worse to better images)
    sorted_zscores = {k: (v-mean)/stdev for k, v in sorted_scores.items()}
    issue_indices = []
    issue_bool = OrderedDict()
    issue_score = []
    for k, v in sorted_zscores.items():
        if v < -2: #more than two standard deviations left from the mean
            issue_indices.append(k)
    for key, val in scores.items(): #ascending indices order for images, label if an image suffer from an issue
        if (val-mean)/stdev < -2:
            issue_bool[key] = 1 
        else:
            issue_bool[key] = 0
        issue_score.append(val)
    return (issue_indices, issue_bool, issue_score, image_names, sorted_scores, sorted_zscores)


def output(dup_indices, images, analyze_scores, verbose = True):
    tests = [bright, prop, entropy]
    issue_names = ["Brightness", "Odd size", "Potential occlusion"]
    issue_dict = {}
    issue_data = {}
    issue_dict["Duplicates"] = dup_indices
    issue_data["Image names"] = image_names
    i = 0
    overall = [] #initialize overall score
    if verbose:
        for x in range(10): #show the first 10 duplicate images (if exists)
            try: 
                img = Image.open(images[dup_indices[x]])
                img.show()
            except:
                break
    for t in tests:
        print(t)
        analysis = analyze_scores(t)
        print(analysis)
        im = analysis[1].keys()
        boolean = list(analysis[1].values())
        issue_indices = analysis[0]
        issue_scores = analysis[2]
        if len(overall) == 0:
            overall = np.array(issue_scores)
        else:   
            overall = overall * np.array(issue_scores) #element wise multiplication with previous scores
        issue_name = issue_names[i]
        i+=1
        issue_dict[issue_name] = issue_indices
        issue_data[issue_name+" issue"] = boolean
        issue_data[issue_name+" score"] = issue_scores
        if verbose:
            for ind in range(10): #show the top 10 issue images (if exists)
                try:
                    img = Image.open(images[issue_indices[ind]])
                    img.show()
                except:
                    break
    issue_data["Overall Score"] = list(overall)
    issue_df = pd.DataFrame(issue_data, index=im)
    return (issue_dict, issue_df)

print(output(dup_indices, image_names, analyze_scores, False))
    


end = time.time()
total_time = end - start
print("\n"+ str(total_time))

    
        