from cleanvision.imagelab import Imagelab
from cleanvision.dataset.folder_dataset import FolderDataset


class TestImagelab:
    def test_list_default_issue_types(self):
        default_issues = Imagelab.list_default_issue_types()
        assert default_issues == [
            "dark",
            "light",
            "odd_aspect_ratio",
            "low_information",
            "exact_duplicates",
            "near_duplicates",
            "blurry",
            "grayscale",
            "odd_size",
        ]

    def test_imagelab_init(self, generate_local_dataset):
        imagelab = Imagelab(data_path=generate_local_dataset)
        assert isinstance(imagelab._dataset, FolderDataset)
        assert len(imagelab._dataset) == 40
        assert imagelab.cleanvision_version
        assert len(imagelab.info["statistics"]) == 0
        assert imagelab.issue_summary.empty
        assert imagelab.issues.empty

    def list_possible_issue_types(self):
        Imagelab.list_possible_issue_types()
