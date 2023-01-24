import pytest

from cleanvision.issue_managers import IssueType
from cleanvision.issue_managers.duplicate_issue_manager import DuplicateIssueManager

exact = IssueType.EXACT_DUPLICATES.value
near = IssueType.NEAR_DUPLICATES.value


class TestDuplicateIssueManager:
    @pytest.fixture
    def issue_manager(self):
        return DuplicateIssueManager(params={})

    def test_set_params(self):
        pass

    def test_get_issue_types_to_compute(self):
        pass

    def test_update_info(self):
        pass

    @pytest.mark.parametrize(
        "before_info, after_info",
        [
            (
                {exact: {"sets": [["im5", "im6"]]}, near: {"sets": [["im1", "im2"]]}},
                {exact: {"sets": [["im5", "im6"]]}, near: {"sets": [["im1", "im2"]]}},
            ),
            (
                {
                    exact: {"sets": [["im1", "im2"], ["im5", "im6"]]},
                    near: {"sets": [["im1", "im2"], ["im3", "im4"]]},
                },
                {
                    exact: {"sets": [["im1", "im2"], ["im5", "im6"]]},
                    near: {"sets": [["im3", "im4"]]},
                },
            ),
        ],
    )
    def test_remove_exact_duplicates_from_near(
        self, before_info, after_info, issue_manager, monkeypatch
    ):
        # Assumes info["exact"] is always populated

        with monkeypatch.context() as m:
            m.setattr(issue_manager, "info", before_info)
            issue_manager._remove_exact_duplicates_from_near()
            assert issue_manager.info == after_info

    @pytest.mark.parametrize(
        "hash_image_mapping, expected_duplicate_sets",
        [
            ({"h1": ["im1"], "h2": ["im2"], "h3": ["im3"]}, []),
            (
                {"h1": ["im1", "im4"], "h2": ["im2", "im5", "im6"], "h3": ["im3"]},
                [["im1", "im4"], ["im2", "im5", "im6"]],
            ),
        ],
    )
    def test_get_duplicate_sets(
        self, hash_image_mapping, expected_duplicate_sets, issue_manager
    ):
        """tests DuplicateIssueManager._get_duplicate_sets(). Cases covered:
        1. No duplicate sets
        2. One or more duplicate sets

        Parameters
        ----------
        hash_image_mapping
        expected_duplicate_sets
        issue_manager
        """
        duplicate_sets = issue_manager._get_duplicate_sets(hash_image_mapping)
        assert len(duplicate_sets) == len(expected_duplicate_sets)
        duplicate_sets.sort()
        expected_duplicate_sets.sort()
        for s1, s2 in zip(duplicate_sets, expected_duplicate_sets):
            assert set(s1) == set(s2)
