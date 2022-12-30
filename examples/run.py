from clean_vision.imagelab import Imagelab

if __name__ == "__main__":
    dataset_path = "../../image_files"

    # Run Imagelab with default settings
    imagelab = Imagelab(dataset_path)
    imagelab.find_issues()
    imagelab.report()

    # Run Imagelab for specific checks
    imagelab = Imagelab(dataset_path)
    issue_types = {"Dark": {}}
    imagelab.find_issues(issue_types)
    imagelab.report()

    # Check for additional types of issues using existing Imagelab
    issue_types = {"Light": {}}
    imagelab.find_issues(issue_types)
    imagelab.report()

    # Run Imagelab for custom thresholds
    imagelab = Imagelab(dataset_path)
    issue_types = {"Dark": {"threshold": 0.2}}
    imagelab.find_issues(issue_types)
    imagelab.report()

    # Customize report
    # Change verbosity
    imagelab = Imagelab(dataset_path)
    imagelab.find_issues()
    imagelab.report(verbosity=3)

    # topk and verbose are conflicting arguments right now
    imagelab = Imagelab(dataset_path)
    imagelab.find_issues()
    # Find top examples suffering from issues that are not present in more than 1% of the dataset
    imagelab.report(num_top_issues=1, max_prevalence=0.01)

    # Visualize
    imagelab.visualize(["Light"], examples_per_issue=8, figsize=(9, 9))
