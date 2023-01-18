from cleanvision.imagelab import Imagelab

if __name__ == "__main__":
    dataset_path = "../../image_files"

    # Run Imagelab with default settings
    imagelab = Imagelab(dataset_path)
    imagelab.list_default_issue_types()
    imagelab.find_issues()
    imagelab.report()

    # Run Imagelab for specific checks
    imagelab = Imagelab(dataset_path)
    issue_types = {"dark": {}}
    imagelab.find_issues(issue_types)
    imagelab.report()

    # Check for additional types of issues using existing Imagelab
    issue_types = {"light": {}, "low_information": {}}
    imagelab.find_issues(issue_types)
    imagelab.report()

    # Check for an issue with a different threshold
    issue_types = {"dark": {"threshold": 0.2}}
    imagelab.find_issues(issue_types)
    imagelab.report()

    # Run imagelab for default issue_type, but override parameters for one or more issues
    imagelab = Imagelab(dataset_path)
    imagelab.find_issues()
    issue_types = {"dark": {"threshold": 0.2}}
    imagelab.find_issues(issue_types)
    imagelab.report()

    # Customize report
    # Change verbosity
    imagelab.report(verbosity=3)

    # Report arg values here will overwrite verbosity defaults
    # Find top examples suffering from issues that are not present in more than 1% of the dataset
    imagelab.report(num_top_issues=1, max_prevalence=0.01)

    # Visualize
    imagelab.visualize(["light"], examples_per_issue=8, figsize=(9, 9))

    # Get stats
    stats = imagelab.info["statistics"]
    print(stats.keys())

    # Run imagelab on custom issue
    imagelab.list_possible_issue_types()
    issue_types = {"custom": {}}
    imagelab.find_issues(issue_types)
    imagelab.report()
