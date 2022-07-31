import time, os, sys
from image_data_quality.image_dataset import ImageDataset

start = time.time()
images = ImageDataset("image_files/")
print("The path is:", images.path)
print("There are ", str(len(images.image_files)), "images in the dataset")
issues = images.find_issues(threshold = None, issues_checked={"Blurry":{}}, verbose=False, num_preview=5) #the dictionary of issues and pandas dataframe of information
images.find_issues(threshold = None, issues_checked={"Brightness":{}}, verbose=False, num_preview=5)
images.find_issues(threshold = None, issues_checked={"Near Duplicates":{}}, verbose=False, num_preview=5)
print(issues)
#print(issues2)
#issues checked make into a new dictionaries, values are keyword arguments (hyperparameter)
end = time.time()
total_time = end - start
print("\n"+ str(total_time))


print(images)
