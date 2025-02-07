from cleanvision import Imagelab

print("Running short output test with verbose=True")
# Specify path to folder containing the image files in your dataset
imagelab = Imagelab(data_path="image_files/", verbose=True)
# Automatically check for a predefined list of issues within your dataset
imagelab.find_issues(verbose=True)

print("Running short output test with verbose=False")
# Specify path to folder containing the image files in your dataset
imagelab = Imagelab(data_path="image_files/", verbose=False)
# Automatically check for a predefined list of issues within your dataset
imagelab.find_issues(verbose=False)
