from PIL import Image

from cleanvision.utils.utils import get_filepaths, update_df
import pandas as pd
import numpy as np
import pytest


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
    def test_update_df(self, num_cols_old, num_cols_new, num_common_cols):
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
        print("old df ")
        print(old_df)
        print("new df")
        print(new_df)

        # testing overrite = True
        updated_df = update_df(old_df, new_df, overwrite=True)
        print("updated df with overwrite = True")
        print(updated_df)
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
        assert num_common_cols == 0 or updated_df[common_col_names].equals(
            new_df[common_col_names]
        )

        # testing overrite = False
        updated_df = update_df(old_df, new_df, overwrite=False)
        print("updated df with overwrite = False")
        print(updated_df)
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
        assert num_common_cols == 0 or updated_df[common_col_names].equals(
            old_df[common_col_names]
        )
