import statistics, os, glob
from collections import OrderedDict
from PIL import Image

types = ["*.jpg", "*.jpeg", "*.gif", "*.jp2", "*.TIFF", "*.WebP","*.PNG"] #filetypes supported by PIL

def get_sorted_images(path): #Question: should this not be its own function but instead just carried out in __init__ of our Class?
    '''
    Used in initialization of ImageDataset Class
    Sorts image files based on image filenames numerically and alphabetically
    
     
    Parameters
    ----------
    path: string (an attribute of ImageDataset Class)
    a string represening the current working directory
    
    
    Returns
    -------
    sorted_names: list
    a list of image filenames sorted numerically and alphabetically
    '''
    raw_images = []
    pathlen = len(path)
    for type in types:
        filetype = glob.glob(os.path.join(path, type))
        if filetype == []:
            continue
        raw_images += filetype
    raw_names = []
    for r in raw_images: 
        raw_names.append(r[pathlen+1:]) #extract image name
    sorted_names = sorted(raw_names)#sort image names alphabetically and numerically
    return sorted_names 

def analyze_scores(scores): 
    '''
    Analyzes the scores for a given issue check, 
    including computing the z-scores (where 2 standard deviations left of mean is considered as significant)
    and sorting image indices based on severity of issue
     
    Parameters
    ----------
    scores: list 
    a list of scores for a particular issue check ordered by image order (index in list corresponds to image index)
    
    Returns
    -------
    a tuple: (issue_indices, issue_bool)
    
    issue_indices: list 
    a list of images indices suffering from the given issue ordered by severity (high to low)
    
    issue_bool: dict
    a dictionary where keys are image indices in ascending order, and respective values are binary integers 
    1 if the image suffers from the given issue
    0 otherwise
    '''
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores)
    scores_dict = {} #stores index and scores
    for i, val in enumerate(scores):
        scores_dict[i] = val
    sorted_scores = {k: v for k, v in sorted(scores_dict.items(), key=lambda item: item[1])} #sort scores from low to high (worse to better images)
    sorted_zscores = {k: (v-mean)/stdev for k, v in sorted_scores.items()}
    issue_indices = [] #high to low severity
    issue_bool = OrderedDict() #ascending image indices, boolean to denote if issue present
    #issue_score = [] #ascending image indices, list of scores 
    for k1, v1 in sorted_zscores.items():
        if v1 < -2: #more than two standard deviations left from the mean
            issue_indices.append(k1)
    for k2, v2 in scores_dict.items(): #ascending indices order for images, label if an image suffer from an issue
        if (v2-mean)/stdev < -2:
            issue_bool[k2] = 1 
        else:
            issue_bool[k2] = 0
    return (issue_indices, issue_bool)


