from clean_vision.imagelab import Imagelab

if __name__ == "__main__":
    dataset_path = "../../image_files"

    # Run Imagelab with default settings
    imagelab = Imagelab(dataset_path)
    imagelab.find_issues()
    imagelab.report()

    # Run Imagelab for specific checks
    imagelab = Imagelab(dataset_path)
    issue_types = {"Dark": None}
    imagelab.find_issues(issue_types)
    imagelab.report()

    # Check for additional types of issues using existing Imagelab
    issue_types = {"Light": None}
    imagelab.find_issues(issue_types)
    imagelab.report()

    # Run Imagelab for custom thresholds
    imagelab = Imagelab(dataset_path)
    issue_types = {"Dark": 0.2}
    imagelab.find_issues(issue_types)
    imagelab.report()

    # Customize report
    # topk and verbose are conflicting arguments right now
    imagelab = Imagelab(dataset_path)
    imagelab.find_issues()
    # Find top examples suffering from issues that are not present in more than 1% of the dataset
    imagelab.report(topk=1, max_prevalence=1)

    # Visualize
    imagelab = Imagelab(dataset_path)
    imagelab.find_issues()
    imagelab.visualize(["Light"], num_images_per_issue=8, figsize=(9, 9))
