""" Example use of this library to check a collection of (unlabeled) images for various issues. 
	First make sure you have placed your images in the path: image_files/
"""
import sys

from image_data_quality.image_dataset import Imagelab

if __name__ == "__main__":
    path_to_images = sys.argv[1]  # folder where your image files live

    print(f"Loading images from {path_to_images}")
    imagelab = Imagelab(path_to_images, thumbnail_size=(128, 128))
    print(f"Number of images: {str(len(imagelab.image_files))}") #todo implement get_num_images

    issue_types = ["Blurred"]
    threshold = 0.5
    # issue_specific_thresholds = {
    #     "Duplicated": threshold,
    #     "DarkImages": 0.08,
    #     "LightImages": 0.05,
    #     "AspectRatio": 0.45,
    #     "Blurred": 0.16,
    #     "Entropy": threshold,
    #     "NearDuplicates": threshold,
    #     "Grayscale": threshold,
    #     "HotPixels": threshold,
    # }

    issues = imagelab.evaluate(issue_types=["DarkImages"])
    imagelab.aggregate()
    imagelab.summary()
    imagelab.visualize(5)

    """
    # Here's how to instead check for one particular issue, with some optional configurations overriden from their defaults):
    issues = imagelab.find_issues(threshold = None, issues_checked={"Near Duplicates":{"hash_size":8}}, verbose=False, num_preview=1) #the dictionary of issues and pandas dataframe of information
    
    # TODO: add code here showing how to view just the images with a particular type of issue once `issues` has been computed
    # Issues checked make into a new dictionaries, values are keyword arguments (hyperparameter)
    print(imagelab.misc_info['Near Duplicate Image Groups'])
    
    
    # Another variant:
    issues = imagelab.find_issues(threshold = None, issues_checked=None, verbose=False, num_preview=0)
    """

    # print(issues)
    #
    # print(imagelab)
    #
    # print(imagelab.issue_info)
    #
    # print(imagelab.misc_info['Near Duplicate Image Groups'])

    # print(imagelab.misc_info)
    #
    # print('FINAL ISSUE SCORES', imagelab.issue_scores)
