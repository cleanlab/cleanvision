import math, hashlib, os, glob
from PIL import ImageStat

types = ["*.jpg", "*.jpeg", "*.gif", "*.jp2", "*.TIFF", "*.WebP","*.PNG"] #filetypes supported by PIL

def sorted_images(path): #Question: should this not be its own function but instead just carried out in __init__ of our Class?
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

def brightness_check(img):
    '''
    Calculates the brightness score for a given image
    
     
    Parameters
    ----------
    img: PIL image
    a PIL image object for which analysis the brightness score is calculated
    
    
    Returns
    -------
    bright_score: float
    a float between 0 and 1 representing if the image suffers from being too bright or too dark
    a lower number means a more severe issue
    '''
    stat = ImageStat.Stat(img)
    try:
        r,g,b = stat.mean
    except:
        print(f"WARNING: {img} does not have just r, g, b values")
    cur_bright = (math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)))/255
    bright_score = min(cur_bright, 1-cur_bright) #too bright or too dark
    return bright_score
    
def prop_check(img):
    '''
    Calculates the proportions score for a given image
    
     
    Parameters
    ----------
    img: PIL image
    a PIL image object for which analysis the brightness score is calculated
    
    
    Returns
    -------
    prop_score: float
    a float between 0 and 1 representing if the image suffers from being having an odd size
    a lower number means a more severe issue
    '''
    width, height = img.size
    prop_score = min(width/height, height/width) #consider extreme shapes
    return prop_score
        
def entropy_check(img):
    '''
    Calculates the entropy score for a given image
    
     
    Parameters
    ----------
    img: PIL image
    a PIL image object for which analysis the brightness score is calculated
    
    
    Returns
    -------
    entropy_score: float
    a float between 0 and 1 representing the entropy of an image
    a lower number means potential occlusion
    '''
    entropy_score = img.entropy()/10 
    return entropy_score

def find_dup(img, image_name, count, dup_indices = [], hashes = set(), dup_dict = {}):
    '''
    Updates hash information for the set of images to find duplicates
    
     
    Parameters
    ----------
    img: PIL image
    a PIL image object for which analysis the brightness score is calculated
    
    image_name: string
    a string representing the image name
    
    count: int
    a integer representing the current image index in the dataset
    
    dup_indices: list
    a list representing the image indices that are duplicates
    
    hashes: set
    a set recording the hashes of all images seen so far
    
    dup_dict: dict
    a dictionary where keys are strings of hashes and values are a list of image names that hashed to this hash
    
    Returns
    -------
    a tuple: (dup_indices, hashes, dup_dict)
    
    a tuple of the parameters updated with new information given by img
    
    '''
    cur_hash = hashlib.md5(img.tobytes()).hexdigest()
    if cur_hash in hashes:
        dup_indices.append(count-1) #append image index
        dup_dict[cur_hash].append(image_name)
    else:
        hashes.add(cur_hash)
        dup_dict[cur_hash] = [image_name]
    return (dup_indices, hashes, dup_dict)