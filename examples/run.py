from clean_vision.imagelab import Imagelab

if __name__ == "__main__":
    dataset_path = "../../image_files"

    imagelab = Imagelab(dataset_path)
    issue_types = {"Dark": None}

    imagelab.find_issues()
    imagelab.report()
