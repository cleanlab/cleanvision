import os

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

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
        imagelab.find_issues()

        save_folder = tmp_path / "T_save_folder/"
        imagelab.save(save_folder)

        loaded_imagelab = Imagelab.load(save_folder)
        assert loaded_imagelab is not None
        assert_frame_equal(loaded_imagelab.issues, imagelab.issues)
        assert_frame_equal(loaded_imagelab.issue_summary, imagelab.issue_summary)
        self.compare_dict(loaded_imagelab.info, imagelab.info)

    def compare_dict(self, a, b):
        assert len(a) == len(b)
        for k, v in a.items():
            print(k)
            assert k in b
            if isinstance(v, dict):
                self.compare_dict(v, b[k])
            elif isinstance(v, pd.DataFrame):
                assert_frame_equal(v, b[k])
            elif isinstance(v, pd.Series):
                assert_series_equal(v, b[k])
            else:
                assert v == b[k]

    def test_load_file_does_not_exist(self, generate_local_dataset, tmp_path):
        save_folder = tmp_path / "T_save_folder/"
        with pytest.raises(ValueError):
            Imagelab.load(save_folder)

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
