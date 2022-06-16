import math, hashlib
from PIL import ImageStat


def check_brightness(img):
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
        r,g,b = stat.mean[:3]
        print(f"WARNING: {img} does not have just r, g, b values")
    cur_bright = (math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2)))/255
    bright_score = min(cur_bright, 1-cur_bright) #too bright or too dark
    return bright_score
    
def check_odd_size(img):
    '''
    Calculates the proportions score for a given image to find odd image sizes
    
     
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
        
def check_entropy(img):
    '''
    Calculates the entropy score for a given image to find potentially occluded images
    
     
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

def check_duplicated(img, image_name, count, dup_indices = [], hashes = set(), dup_dict = {}):
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