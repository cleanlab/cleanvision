import pytest

from cleanvision.issue_managers.duplicate_issue_manager import DuplicateIssueManager


class TestDuplicateIssueManager:
    @pytest.fixture
    def issue_manager(self):
        return DuplicateIssueManager(params={})
