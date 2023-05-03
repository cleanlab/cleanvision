import os

import pytest

from cleanvision.imagelab import Imagelab


class TestImagelabSaveLoad:
    def test_save(self, generate_local_dataset, tmp_path):
        imagelab = Imagelab(data_path=generate_local_dataset)
        save_folder = tmp_path / "T_save_folder/"
        imagelab.save(save_folder)
        assert os.path.exists(save_folder / "imagelab.pkl")
        assert os.path.exists(save_folder / "issues.csv")
        assert os.path.exists(save_folder / "issue_summary.csv")

    @pytest.mark.parametrize(
        "force",
        [True, False],
        ids=["overwrite", "do not overwrite"],
    )
    def test_force_save(self, generate_local_dataset, tmp_path, force):
        save_folder = tmp_path / "T_save_folder/"
        save_folder.mkdir()
        imagelab = Imagelab(data_path=generate_local_dataset)
        if force:
            imagelab.save(save_folder, force=force)
            assert os.path.exists(save_folder / "imagelab.pkl")
            assert os.path.exists(save_folder / "issues.csv")
            assert os.path.exists(save_folder / "issue_summary.csv")
        else:
            with pytest.raises(FileExistsError):
                imagelab.save(save_folder, force=force)
