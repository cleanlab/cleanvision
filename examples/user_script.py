import time, os, sys
from image_data_quality.image_dataset import ImageDataset

start = time.time()
imgset = ImageDataset()

print("The path is:", imgset.path) 
print("There are ", str(len(imgset.image_files)), "images in the dataset")
print(imgset.audit_images(False, 3)) #returns the dictionary of issues and pandas dataframe of information
print(imgset.issue_info)

end = time.time()
total_time = end - start
print("\n"+ str(total_time))

