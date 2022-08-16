import time, os, sys
from image_data_quality.image_dataset import ImageDataset

start = time.time()
images = ImageDataset("image_files/")
print("The path is:", images.path)
print("There are ", str(len(images.image_files)), "images in the dataset")
issues = images.find_issues(threshold = None, issues_checked=None, verbose=False, num_preview=0) #the dictionary of issues and pandas dataframe of information
#print(images.misc_info['Near Duplicate Image Groups'])
print(issues)
print("HERE", images.issues_checked)
#issues checked make into a new dictionaries, values are keyword arguments (hyperparameter)
end = time.time()
total_time = end - start
print("\n"+ str(total_time))


print(images)
