import pytest
from PIL import Image

from cleanvision.issue_managers import IssueType
from cleanvision.issue_managers.duplicate_issue_manager import (
    DuplicateIssueManager,
    get_hash,
)

EXACT = IssueType.EXACT_DUPLICATES.value
NEAR = IssueType.NEAR_DUPLICATES.value


class TestDuplicateIssueManager:
    @pytest.fixture
    def issue_manager(self):
        return DuplicateIssueManager()

    @pytest.fixture
    def set_default_params(self, issue_manager, monkeypatch):
        """Set default params for image property issue types"""

        def mock_get_default_params():
            return {
                EXACT: {"hash_type": "md5"},
                NEAR: {"hash_type": "whash", "hash_size": 8},
            }

        monkeypatch.setattr(issue_manager, "params", mock_get_default_params())

    @pytest.mark.usefixtures("set_default_params")
    @pytest.mark.parametrize(
        "params,expected_params",
        [
            (
                {EXACT: {}, NEAR: {"hash_size": 16}},
                {
                    EXACT: {"hash_type": "md5"},
                    NEAR: {"hash_type": "whash", "hash_size": 16},
                },
            )
        ],
        ids=["use both default and specified params as necessary"],
    )
    def test_set_params(self, params, expected_params, issue_manager):
        """Tests DuplicateIssueManager.set_params(). Cases covered:
        1. If no params are specified for an issue_type, default params are used
        2. If params are specified, those specific params are updated, for the remaining ones default values are used
        """
        issue_manager.update_params(params)
        assert issue_manager.params == expected_params

    @pytest.mark.parametrize(
        "issue_types, imagelab_info, expected_to_compute",
        [
            ([EXACT], {}, [EXACT]),
            ([NEAR], {}, [EXACT, NEAR]),
            ([EXACT, NEAR], {}, [EXACT, NEAR]),
            ([EXACT], {EXACT: {"sets": []}}, []),
            ([NEAR], {EXACT: {"sets": []}, NEAR: {"sets": []}}, [NEAR]),
        ],
        ids=[
            "Only exact, compute exact",
            "Only near, compute both",
            "Both, compute both",
            "Reuse exact",
            "Compute near even if precomputed",
        ],
    )
    def test_get_issue_types_to_compute(
        self, issue_types, imagelab_info, expected_to_compute, issue_manager
    ):
        """Tests DuplicateIssueManager._get_issue_types_to_compute().
        Always compute near duplicates if specified in issue_types
        Always compute exact if not present in imagelab_info as it is required both by exact and near
        Reuse exact duplicates if present in imagelab_info
        """
        to_compute = issue_manager._get_issue_types_to_compute(
            issue_types, imagelab_info
        )
        assert set(to_compute) == set(expected_to_compute)

    @pytest.mark.parametrize(
        "issue_types, issue_type_hash_mapping, imagelab_info, expected_info",
        [
            ([EXACT], {EXACT: {}}, {}, {EXACT: {"sets": []}}),
            ([EXACT], {}, {EXACT: {"sets": []}}, {EXACT: {"sets": []}}),
            (
                [NEAR],
                {EXACT: {}, NEAR: {}},
                {},
                {EXACT: {"sets": []}, NEAR: {"sets": []}},
            ),
            (
                [NEAR],
                {NEAR: {}},
                {EXACT: {"sets": []}},
                {EXACT: {"sets": []}, NEAR: {"sets": []}},
            ),
            (
                [EXACT, NEAR],
                {NEAR: {"h1": ["im_new1", "im_new2"]}},
                {EXACT: {"sets": []}, NEAR: {"sets": [["im_old1", "im_old2"]]}},
                {EXACT: {"sets": []}, NEAR: {"sets": [["im_new1", "im_new2"]]}},
            ),
        ],
        ids=[
            "Only exact",
            "Only exact, reuse exact",
            "Only near, compute both",
            "Only near, reuse exact",
            "Both exact and near, update near sets",
        ],
    )
    def test_update_info(
        self,
        issue_types,
        issue_type_hash_mapping,
        imagelab_info,
        expected_info,
        issue_manager,
    ):
        """Tests DuplicateIssueManager._update_info(). Cases covered:
        1.  Only exact is present in issue_types
            issue_manager.info should only contain duplicate sets for exact issue_types
            exact sets can be retrieved either from issue_type_hash_mapping or imagelab_info
            test_ids: "Only exact", "Only exact, reuse exact"
        2.  Only near is present in issue_types
            issue_manager.info contains both exact and near duplicate sets, as duplicate sets is used for
            calculating near duplicate sets.
            Near duplicates are always calculated, hence always present in issue_type_hash_mapping
            Exact duplicates can be retrieved either from issue_type_hash_mapping or imagelab_info
            test_ids: "Only near, compute both", "Only near, reuse exact",
        3.  Both exact and near are present in issue_types
            issue_manager.info should contain both exact and near duplicate sets
            Near duplicate sets are always calculated, and updated with new results
            test_id: "Both exact and near, update near sets",
        """
        issue_manager._update_info(issue_types, issue_type_hash_mapping, imagelab_info)
        assert set(issue_types).issubset(issue_manager.info.keys())

    @pytest.mark.parametrize(
        "before_info, after_info",
        [
            (
                {EXACT: {"sets": [["im5", "im6"]]}, NEAR: {"sets": [["im1", "im2"]]}},
                {EXACT: {"sets": [["im5", "im6"]]}, NEAR: {"sets": [["im1", "im2"]]}},
            ),
            (
                {
                    EXACT: {"sets": [["im1", "im2"], ["im5", "im6"], ["im7", "im8"]]},
                    NEAR: {
                        "sets": [["im1", "im2"], ["im3", "im4"], ["im7", "im8", "im9"]]
                    },
                },
                {
                    EXACT: {"sets": [["im1", "im2"], ["im5", "im6"], ["im7", "im8"]]},
                    NEAR: {"sets": [["im3", "im4"], ["im7", "im8", "im9"]]},
                },
            ),
        ],
        ids=["No overlapping sets", "Identical overlapping set between exact and near"],
    )
    def test_remove_exact_duplicates_from_near(
        self, before_info, after_info, issue_manager, monkeypatch
    ):
        """Tests DuplicateIssueManager._remove_exact_duplicates_from_near().
        Only identical exact duplicate sets are removed from near duplicates,
        partial sets are not considered for removal
        """
        monkeypatch.setattr(issue_manager, "info", before_info)
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
        ids=["No duplicate sets", "Multiple duplicate sets"],
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


def test_get_hash():
    img = Image.new("RGB", (200, 200), (255, 0, 0))
    hash = get_hash(img, {"hash_type": "md5"})
    assert hash is not None

    hash = get_hash(img, {"hash_type": "whash", "hash_size": 2})
    assert hash is not None

    hash = get_hash(img, {"hash_type": "phash", "hash_size": 2})
    assert hash is not None

    # Test calling function with not a real hash
    with pytest.raises(ValueError, match="not supported"):
        get_hash(img, {"hash_type": "fake_hash"})
