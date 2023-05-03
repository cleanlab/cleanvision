import os

import pytest

import cleanvision
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

    def test_load(self, generate_local_dataset, tmp_path):
        imagelab = Imagelab(data_path=generate_local_dataset)
        save_folder = tmp_path / "T_save_folder/"
        imagelab.save(save_folder)
        imagelab = Imagelab.load(save_folder)
        assert imagelab is not None
        assert imagelab.issues is not None
        assert imagelab.issue_summary is not None

    def test_load_file_does_not_exist(self, generate_local_dataset, tmp_path):
        save_folder = tmp_path / "T_save_folder/"
        with pytest.raises(ValueError):
            imagelab = Imagelab.load(save_folder)

    def test_warning_raised_on_diff_version(
        self, generate_local_dataset, tmp_path, monkeypatch
    ):
        save_folder = tmp_path / "T_save_folder/"
        imagelab = Imagelab(data_path=generate_local_dataset)
        imagelab.save(save_folder)

        monkeypatch.setattr(cleanvision, "__version__", "dummy")

        with pytest.warns(UserWarning) as record:
            imagelab.load(save_folder)

        warning_message = record[0].message.args[0]
        assert (
            "Saved Imagelab was created using different version of cleanvision"
            in warning_message
        )
