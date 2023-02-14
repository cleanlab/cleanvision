import argparse

from cleanvision.imagelab import Imagelab

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrates how to use Imagelab")
    parser.add_argument("--path", type=str, help="path to dataset", required=True)
    args = parser.parse_args()

    dataset_path = args.path

    """
    Example 1

    This example demonstrates the default Imagelab workflow to detect various types of issues in an image dataset.
    """

    imagelab = Imagelab(data_path=dataset_path)  # initalize imagelab
    imagelab.list_default_issue_types()  # list default checks
    imagelab.visualize()  # visualize random images in dataset

    imagelab.find_issues()  # Find issues in the dataset
    imagelab.report()

    print("Summary of all issues checks\n", imagelab.issue_summary.to_markdown())
    imagelab.visualize(
        issue_types=["blurry"], num_images=8
    )  # visualize images that have specific issues

    # Get all images with blurry issue type
    blurry_images = imagelab.issues[
        imagelab.issues["blurry_bool"] == True
    ].index.to_list()
    imagelab.visualize(image_files=blurry_images)  # visualize the given image files

    # Miscellaneous extra information about dataset and its issues
    print(list(imagelab.info.keys()), "\n")
    print(list(imagelab.info["statistics"].keys()))
    print(imagelab.info["statistics"]["brightness"][:10])

    """
    Example 2

    This example demonstrates using Imagelab to:
    1. Check data for specific types of issues
    2. Incrementally detect additional types of issues with  existing Imagelab
    3. Specify nondefault parameter to use when detecting a particular issue type (e.g. a different threshold)
    4. Save and load Imagelab to file
    5. Report only specific issue types
    """

    imagelab = Imagelab(data_path=dataset_path)
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
    imagelab.report(issue_types=issue_types.keys())  # report only specific issues

    """
    Example 3

    This example demonstrates using Imagelab to:
    1. Check for all default issue types, overriding some parameters for a particular issue type from their default values.
    2. Change the verbosity of generated report to see more details
    3. Ignore issues occurring in more than x% of images in the dataset
    4. Increase the size of images in the grid displayed by visualize
    """

    imagelab = Imagelab(data_path=dataset_path)
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
    imagelab.visualize(issue_types=["light"], num_images=8, cell_size=(3, 3))

    """
    Example 4

    This example demonstrates creating your own custom issue and using Imagelab to detect this additional issue type, along with the default set of issues
    """
    # Run imagelab on custom issue
    from custom_issue_manager import CustomIssueManager

    imagelab = Imagelab(data_path=dataset_path)
    issue_name = CustomIssueManager.issue_name
    imagelab.list_possible_issue_types()

    issue_types = {issue_name: {}}
    imagelab.find_issues(issue_types)  # check for custom issue type

    imagelab.find_issues()  # also check for default issue types
    imagelab.report()
