import argparse

from cleanvision.imagelab import Imagelab

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrates how to use Imagelab")
    parser.add_argument("--path", type=str, help="path to dataset", required=True)
    args = parser.parse_args()

    dataset_path = args.path

    """
    Example 1

    This example demonstrates using Imagelab to
    1. Explore the dataset
    2. Find all types of issues
    3. Report top issues found and visualize them
    4. Check Imagelab attributes: imagelab.issue_summary, imagelab.issues, imagelab.info
    """

    imagelab = Imagelab(dataset_path)  # initalize imagelab
    imagelab.list_default_issue_types()  # list default checks
    imagelab.visualize()  # visualize random images in dataset

    imagelab.find_issues()  # Find issues in the dataset
    imagelab.report()

    print("Summary of all issues checks\n", imagelab.issue_summary.to_markdown())
    imagelab.visualize(
        issue_types=["blurry"], examples_per_issue=8
    )  # visualize images that have specific issues

    # Get all images with blurry issue type
    blurry_images = imagelab.issues[
        imagelab.issues["blurry_bool"] == True
    ].index.to_list()
    imagelab.visualize(
        image_files=blurry_images
    )  # visualize the given image files

    # More info on issue checks
    print(list(imagelab.info.keys()), "\n")
    print(list(imagelab.info["statistics"].keys()))
    print(imagelab.info["statistics"]["brightness"][:10])

    """
    Example 2

    This examples demonstrates using Imagelab to
    1. Check for specific issue types
    2. Incrementally running checks for different issue types
    3. Checking for an issue type with a different threshold
    4. Save and load Imagelab
    5. Report specific issue types
    """

    imagelab = Imagelab(dataset_path)
    issue_types = {"near_duplicates": {}}
    imagelab.find_issues(issue_types)
    imagelab.report()
    imagelab.save(
        "./results"
    )  # optional, just included to show how to save/load this as a file

    # Check for additional types of issues using existing Imagelab
    imagelab = Imagelab.load("./results", dataset_path)
    issue_types = {"light": {}, "low_information": {}}
    imagelab.find_issues(issue_types)
    imagelab.report()

    # Check for an issue with a different threshold
    issue_types = {"dark": {"threshold": 0.2}}
    imagelab.find_issues(issue_types)
    imagelab.report(issue_types=issue_types.keys())  # report specific issues

    """
    Example 3

    This examples demonstrates using Imagelab to
    1. Check for all issue types,but override the hyperparameters for an issue type
    2. Change verbosity of report
    3. Filter out issues occurring in more than x% of the dataset
    4. Increase the cell size of image in image grid
    """

    # Run imagelab for default issue_type, but override parameters for one or more issues
    imagelab = Imagelab(dataset_path)
    imagelab.find_issues()
    imagelab.report(["near_duplicates"])

    issue_types = {"near_duplicates": {"hash_type": "phash"}}
    imagelab.find_issues(issue_types)
    imagelab.report(issue_types=issue_types.keys())

    # Customize report and visualize

    # Change verbosity
    imagelab.report(verbosity=3)

    # Report arg values here will overwrite verbosity defaults
    # Find top examples suffering from issues that are not present in more than 1% of the dataset
    imagelab.report(num_top_issues=5, max_prevalence=0.01)

    # Increase cell_size in the grid
    imagelab.visualize(issue_types=["light"], examples_per_issue=8, cell_size=(3, 3))

    """
    Example 4

    This example demonstrates creating your own custom issue and using Imagelab to check for the added issue type
    """
    # Run imagelab on custom issue
    from custom_issue_manager import CustomIssueManager

    imagelab = Imagelab(dataset_path)
    issue_name = CustomIssueManager.issue_name
    imagelab.list_possible_issue_types()
    issue_types = {issue_name: {}}
    imagelab.find_issues(issue_types)
    imagelab.report(issue_types=[issue_name])
