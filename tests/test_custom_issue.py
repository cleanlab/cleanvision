from docs.source.tutorials.custom_issue_manager import CustomIssueManager
from cleanvision import Imagelab


def test_custom_issue_manager(generate_local_dataset, len_dataset):
    imagelab = Imagelab(data_path=generate_local_dataset)
    issue_name = CustomIssueManager.issue_name

    issue_types = {issue_name: {}}
    imagelab.find_issues(issue_types)

    assert set(imagelab.issue_summary["issue_type"].values) == set([issue_name])
    assert len(imagelab.issues) == len_dataset
    assert set(["is_custom_issue", "custom_score"]) == set(imagelab.issues.columns)
    assert set(["statistics", "custom"]) == set(imagelab.info.keys())
