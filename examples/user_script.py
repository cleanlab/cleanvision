import time, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import importlib  
# image_data_quality = importlib.import_module("image-data-quality")
# from image_data_quality.image_dataset import ImageDataset
from image_data_quality.image_dataset import ImageDataset
#Question: can access from examples folder?
#progress bars


start = time.time()
imgset = ImageDataset()

print("The path is:", imgset.path) #Question: How to stop this from triggering the progress bar?
print("There are ", str(len(imgset.image_files)), "images in the dataset")
print(imgset.audit_images(False)) #returns the dictionary of issues and pandas dataframe of information


end = time.time()
total_time = end - start
print("\n"+ str(total_time))

