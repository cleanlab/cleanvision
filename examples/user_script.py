import time, os, sys
import imagehash
from image_data_quality.image_dataset import ImageDataset

start = time.time()
imgset = ImageDataset("./examples/")
kwargs = {"Near Duplicates" : {"hashtype": imagehash.whash}}
print("The path is:", imgset.path) 
print("There are ", str(len(imgset.image_files)), "images in the dataset")
issues = imgset.find_issues(verbose=False, num_preview=15, threshold = 5, issues_checked=None, **kwargs) #the dictionary of issues and pandas dataframe of information
print(issues)

end = time.time()
total_time = end - start
print("\n"+ str(total_time))
