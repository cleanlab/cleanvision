""" Example use of this library to check a collection of (unlabeled) images for various issues. 
	First make sure you have placed your images in the path: image_files/
""" 


import time, os, sys
from image_data_quality.image_dataset import ImageDataset

start = time.time()

path_to_images = "image_files/"  # folder where your image files live
images = ImageDataset(path_to_images, thumbnail_size = (128,128))
print("The path is:", images.path)
print("There are ", str(len(images.image_files)), "images in the dataset")

issues = images.find_issues()

"""
# Here's how to instead check for one particular issue, with some optional configurations overriden from their defaults):
issues = images.find_issues(threshold = None, issues_checked={"Near Duplicates":{"hash_size":8}}, verbose=False, num_preview=0) #the dictionary of issues and pandas dataframe of information

# TODO: add code here showing how to view just the images with a particular type of issue once `issues` has been computed
# Issues checked make into a new dictionaries, values are keyword arguments (hyperparameter)
print(images.misc_info['Near Duplicate Image Groups'])
"""

print(issues)

end = time.time()
total_time = end - start
print("\n"+ str(total_time))

print(images)