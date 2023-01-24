import pytest

from cleanvision.issue_managers import IssueType
from cleanvision.issue_managers.image_property_issue_manager import (
    ImagePropertyIssueManager,
)

dark = IssueType.DARK.value
blurry = IssueType.BLURRY.value
light = IssueType.LIGHT.value


class TestImagePropertyIssueManager:
    @pytest.fixture
    def issue_manager(self, monkeypatch):
        def mock_init(*args, **kwargs):
            pass

        monkeypatch.setattr(ImagePropertyIssueManager, "__init__", mock_init)
        issue_manager = ImagePropertyIssueManager(params={})
        return issue_manager

    @pytest.fixture
    def set_default_params(self, issue_manager, monkeypatch):
        """Set default params for image property issue types"""

        def mock_get_default_params():
            return {
                dark: {"threshold": 0.22},
                blurry: {"threshold": 0.3, "normalizing_factor": 0.001},
                light: {"threshold": 0.05},
            }

        monkeypatch.setattr(
            issue_manager, "get_default_params", mock_get_default_params
        )

    @pytest.mark.usefixtures("set_default_params")
    @pytest.mark.parametrize(
        "params,expected_params",
        [
            (
                {dark: {}, blurry: {"threshold": 0.4}},
                {
                    dark: {"threshold": 0.22},
                    blurry: {
                        "threshold": 0.4,
                        "normalizing_factor": 0.001,
                    },
                    light: {"threshold": 0.05},
                },
            )
        ],
    )
    def test_set_params(self, params, expected_params, issue_manager):
        """Tests image_property_issue_manager.set_params() method for following cases:
        1. Set default parameters when no parameters are specified
        2. Update default parameters with given parameters and preserv default values for remaining parameters
        3. After setting self.params only contains issue_types specified in the input params and not all default issue_types

        Assumes len(params) > 0

        Parameters
        ----------
        params: input to image_property_issue_manager.set_params()
        expected_params: expected image_property_issue_manager.params after function call
        issue_manager: instance of ImagePropertyIssueManager

        """
        issue_manager.set_params(params)
        assert issue_manager.params == expected_params

    @pytest.mark.parametrize(
        "issue_types, imagelab_info, expected_defer_set",
        [
            ([dark, light, blurry], {"statistics": {}}, {light}),
            ([dark, light, blurry], {"statistics": {"brightness": []}}, {dark, light}),
        ],
    )
    def test_get_defer_set(
        self, issue_types, imagelab_info, expected_defer_set, issue_manager
    ):
        """Tests image_property_issue_manager._get_defer_set(). Cases covered:
        1. If two issue types use the same image property, skip one
        2. If properties issue types use exist in imagelab.info, re-use them
        Parameters
        ----------
        issue_types: Given list of issue types
        imagelab_info: imagelab info parameter
        defer_set: expected defer set
        issue_manager: instance of ImagePropertyIssueManager

        Returns
        -------

        """
        defer_set = issue_manager._get_defer_set(issue_types, imagelab_info)
        assert defer_set == expected_defer_set
