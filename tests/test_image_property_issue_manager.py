import pytest

from cleanvision.issue_managers import IssueType
from cleanvision.issue_managers.image_property_issue_manager import (
    ImagePropertyIssueManager,
)

DARK = IssueType.DARK.value
BLURRY = IssueType.BLURRY.value
LIGHT = IssueType.LIGHT.value


class TestImagePropertyIssueManager:
    @pytest.fixture()
    def issue_manager(self, monkeypatch):
        def mock_init(*args, **kwargs):
            pass

        monkeypatch.setattr(ImagePropertyIssueManager, "__init__", mock_init)
        return ImagePropertyIssueManager(params={})

    @pytest.fixture
    def set_default_params(self, issue_manager, monkeypatch):
        """Set default params for image property issue types"""

        def mock_get_default_params():
            return {
                DARK: {"threshold": 0.22},
                BLURRY: {"threshold": 0.3, "normalizing_factor": 0.001},
                LIGHT: {"threshold": 0.05},
            }

        monkeypatch.setattr(
            issue_manager, "get_default_params", mock_get_default_params
        )

    @pytest.mark.usefixtures("set_default_params")
    @pytest.mark.parametrize(
        "params,expected_params",
        [
            (
                {DARK: {}, BLURRY: {"threshold": 0.4}},
                {
                    DARK: {"threshold": 0.22},
                    BLURRY: {
                        "threshold": 0.4,
                        "normalizing_factor": 0.001,
                    },
                    LIGHT: {"threshold": 0.05},
                },
            )
        ],
        ids=["use both default and specified params as necessary"],
    )
    def test_set_params(self, params, expected_params, issue_manager):
        """Tests image_property_issue_manager.set_params() method for following cases:
        1. Set default parameters when no parameters are specified
        2. Update default parameters with given parameters and preserve default values for remaining parameters


        Assumes len(params) > 0

        Parameters
        ----------
        params: input to image_property_issue_manager.set_params()
        expected_params: expected image_property_issue_manager.params after function call
        issue_manager: instance of ImagePropertyIssueManager

        """
        assert not hasattr(issue_manager, "params")
        issue_manager.set_params(params)
        assert issue_manager.params == expected_params

    @pytest.fixture
    def set_image_properties(self, issue_manager, monkeypatch):
        class MockImageProperty:
            name = None

            def __init__(self, name):
                self.name = name

        image_properties = {
            DARK: MockImageProperty("brightness"),
            LIGHT: MockImageProperty("brightness"),
            BLURRY: MockImageProperty("blurriness"),
        }
        monkeypatch.setattr(
            issue_manager, "image_properties", image_properties, raising=False
        )

    @pytest.mark.usefixtures("set_image_properties")
    @pytest.mark.parametrize(
        "issue_types, imagelab_info, expected_defer_set",
        [
            ([DARK, LIGHT, BLURRY], {"statistics": {}}, {LIGHT}),
            ([DARK, LIGHT, BLURRY], {"statistics": {"brightness": []}}, {DARK, LIGHT}),
        ],
        ids=[
            "exclude issue types using same underlying property",
            "exclude issue types with property precomputed",
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
