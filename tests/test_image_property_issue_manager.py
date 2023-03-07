import pandas as pd
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
    def issue_manager(self):
        return ImagePropertyIssueManager()

    @pytest.fixture
    def set_default_params(self, issue_manager, monkeypatch):
        """Set default params for image property issue types"""

        def mock_get_default_params():
            return {
                DARK: {"threshold": 0.22},
                BLURRY: {"threshold": 0.3, "normalizing_factor": 0.001},
                LIGHT: {"threshold": 0.05},
            }

        monkeypatch.setattr(issue_manager, "params", mock_get_default_params())

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
        issue_manager.update_params(params)
        assert issue_manager.params == expected_params

    @pytest.mark.parametrize(
        "issue_types, agg_computations, expected_defer_set",
        [
            ([DARK, LIGHT, BLURRY], pd.DataFrame(), {LIGHT}),
            (
                [DARK, LIGHT, BLURRY],
                pd.DataFrame(columns=["brightness_perc_99", "brightness_perc_5"]),
                {DARK, LIGHT},
            ),
        ],
        ids=[
            "exclude issue types using same underlying property",
            "exclude issue types with property precomputed",
        ],
    )
    def test_get_defer_set(
        self, issue_types, agg_computations, expected_defer_set, issue_manager
    ):
        """Tests image_property_issue_manager._get_defer_set(). Cases covered:
        1. If two issue types use the same image property, skip one
        2. If properties issue types use exist in imagelab.info, re-use them
        Parameters
        ----------
        issue_types: Given list of issue types
        agg_computations: This dataframe contains all computed properties like blurriness, brightness
            as columns for each image that are required for computing issue scores.
        expected_defer_set: expected defer set
        issue_manager: instance of ImagePropertyIssueManager

        Returns
        -------

        """
        defer_set = issue_manager._get_defer_set(issue_types, agg_computations)
        assert defer_set == expected_defer_set
