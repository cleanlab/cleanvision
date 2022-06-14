import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from issue_checks import brightness_check, prop_check, entropy_check, dup_check
from utils import analyze_scores, get_sorted_images

possible_issues= {"Duplicates": dup_check, "Brightness": brightness_check, "Odd size": prop_check,  "Potential occlusion": entropy_check} #Question: could this be defined here?

class ImageDataset:
    def __init__(self, path = None, image_files = None, thumbnail_size = None, checks = None):
        if path is None:
            self.path = os.getcwd()
        if image_files is None:
            self.image_files = get_sorted_images(self.path)
        if thumbnail_size is None:
            self.thumbnail_size = (128, 128)
        #TODO: revisit default value for thumbnail_size
        if checks is None: #defaults to run all checks
            self.checks = list(possible_issues.keys())
        self.issue_info = {} #key: issue name string, value: list of issue scores (index in list corresponds to image index)
        self.misc_info = {} #key: misc info name string, value: intuitive data structure containing that info
        
    def audit_images(self, verbose = True):
        '''
        Audits self.image_files 
        For issue checks performed on each image (i.e. brightness, odd size, potential occlusion)
            for each image, compute the score for each check 
        For issue checks depending on entire image dataset (i.e. duplicates)
            maintain data structures storing info on the entire image dataset
            for each image, take these data structures as input and update accordingly
        calls analyze_scores to perform analysis and obtain data for output
        
         
        Parameters
        ----------
        verbose: bool (defaults to True)
        a boolean variable where iff True, show a subset of images (<= 10) with issues 
        
        
        Returns
        -------
        a tuple: (issue_dict, issue_df)
        
        issue_dict: dict
        a dictionary where keys are string names of issue checks
        and respective values are a list of images indices suffering from the given issue ordered by severity (high to low)
        
        issue_df: pd.DataFrame 
        A pandas dataframe where each row represents a image index 
        each column represents a property of the image
        
        '''
        count = 0
        check_dup = False #flag variable for checking duplicates 
        for image_name in tqdm(self.image_files):
            img = Image.open(image_name)
            img.thumbnail(self.thumbnail_size) 
            #Question: would it be a good idea to keep a set of all the checks that require info from the entire dataset and use this structure?
            for c in self.checks: #run each check for each image
                if c == "Duplicates":
                    check_dup = True
                    dup_check_call = dup_check(img, image_name, count) #Question: is it bad to ask later calls of this function to reference the variables updated below?
                    dup_indices = dup_check_call[0]
                    hashes = dup_check_call[1]
                    dup_dict = dup_check_call[2] #keys are hashes, values are images with that hash
                else:
                    self.issue_info.setdefault(c,[]).append(possible_issues[c](img))
            count += 1
        if check_dup:
            self.issue_info["Duplicates"] = dup_indices
        #Prepares for output
        issue_dict = {}
        issue_data = {}
        overall_scores = [] #product of scores of all checks
        issue_dict["Duplicates"] = dup_indices
        issue_data["Image names"] = self.image_files #Question: is the column label in the right location?
        if verbose:
            print("Here are some duplicate images")
            for x in range(10): #show the first 10 duplicate images (if exists)
                try: 
                    img = Image.open(self.image_files[dup_indices[x]])
                    img.show()
                except:
                    break
        for c1 in self.checks:
            if c1 != "Duplicates":
                analysis = analyze_scores(self.issue_info[c1])
                issue_indices = analysis[0]
                im = analysis[1].keys() #list of ascending image indices
                boolean = list(analysis[1].values())
                issue_scores = self.issue_info[c1]
                if len(overall_scores) == 0:
                    overall_scores = np.array(issue_scores)
                else:   
                    overall_scores = overall_scores * np.array(issue_scores) #element wise multiplication with previous scores
                issue_dict[c1] = issue_indices
                issue_data[c1 +" issue"] = boolean
                issue_data[c1 +" score"] = issue_scores
                if verbose:
                    print("These images have", c1, "issue")
                    for ind in range(10): #show the top 10 issue images (if exists)
                        try:
                            img = Image.open(self.image_files[issue_indices[ind]])
                            img.show()
                        except:
                            break
        issue_data["Overall Score"] = list(overall_scores)
        issue_df = pd.DataFrame(issue_data, index=im)
        #Analysis for misc_info
        dup_tups = []
        for v in dup_dict.values():
            if len(v)>1:
                dup_tups.append(tuple(v))
        self.misc_info["Duplicate Image Groups"] = dup_tups
        return (issue_dict, issue_df)
       
        
        
        
    