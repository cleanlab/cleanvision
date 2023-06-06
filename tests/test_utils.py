from PIL import Image

from cleanvision.utils.utils import get_filepaths, update_df, get_max_n_jobs
import cleanvision
import pandas as pd
import multiprocessing
import numpy as np
import pytest
import psutil


class TestUtils:
    def test_get_filepaths(self, tmp_path):
        img = Image.new("L", (100, 100))
        extensions = [".png", ".jpeg", ".gif", ".jpg"]
        num_names = 2
        for i in range(num_names):
            for ext in extensions:
                p = tmp_path / f"img_{i}{ext}"
                img.save(p)
        filepaths = get_filepaths(tmp_path)
        print(filepaths)
        assert len(filepaths) == len(extensions) * num_names

    @pytest.mark.parametrize(
        "num_cols_old, num_cols_new, num_common_cols",
        [(4, 5, 4), (4, 4, 2), (4, 4, 0)],
        ids=[
            "all columns are common",
            "some columns are common",
            "no columns are common",
        ],
    )
    @pytest.mark.parametrize(
        "overwrite", [True, False], ids=["overwrite=True", "overwrite=False"]
    )
    def test_update_df(self, num_cols_old, num_cols_new, num_common_cols, overwrite):
        num_rows = 5

        num_only_old_cols = num_cols_old - num_common_cols
        num_only_new_cols = num_cols_new - num_common_cols

        # generating old_df
        only_old_col_names = ["old" + str(x) for x in range(num_only_old_cols)]
        old_df = pd.DataFrame(
            np.random.rand(num_rows, num_only_old_cols), columns=only_old_col_names
        )

        # generating new_df
        only_new_col_names = ["new" + str(x) for x in range(num_only_new_cols)]
        new_df = pd.DataFrame(
            np.random.rand(num_rows, num_only_new_cols), columns=only_new_col_names
        )

        # adding common cols
        common_col_names = ["com" + str(x) for x in range(num_common_cols)]
        old_df[common_col_names] = np.random.rand(num_rows, num_common_cols)
        new_df[common_col_names] = np.random.rand(num_rows, num_common_cols)

        updated_df = update_df(old_df, new_df, overwrite=overwrite)
        assert updated_df.shape == (
            num_rows,
            num_cols_old + num_cols_new - num_common_cols,
        )
        assert num_only_old_cols == 0 or updated_df[only_old_col_names].equals(
            old_df[only_old_col_names]
        )
        assert num_only_new_cols == 0 or updated_df[only_new_col_names].equals(
            new_df[only_new_col_names]
        )

        if overwrite:
            assert num_common_cols == 0 or updated_df[common_col_names].equals(
                new_df[common_col_names]
            )
        else:
            assert num_common_cols == 0 or updated_df[common_col_names].equals(
                old_df[common_col_names]
            )

    def test_get_max_n_jobs(self, monkeypatch):
        def mock_physical_cores(*args, **kwargs):
            return 4

        def mock_logical_cores(*args, **kwargs):
            return 8

        monkeypatch.setattr(psutil, "cpu_count", mock_physical_cores)
        monkeypatch.setattr(multiprocessing, "cpu_count", mock_logical_cores)

        monkeypatch.setattr(cleanvision.utils.utils, "PSUTIL_EXISTS", True)
        njobs = get_max_n_jobs()
        assert njobs == 4

        monkeypatch.setattr(cleanvision.utils.utils, "PSUTIL_EXISTS", False)
        njobs = get_max_n_jobs()
        assert njobs == 8
