import pytest
from PIL import Image

from cleanvision import Imagelab


@pytest.fixture
def image_paths(tmp_path):
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), (90, 120, 150)).save(image)
    return [str(image), str(tmp_path / "missing.png")]


def test_storage_options_are_preserved(image_paths):
    options = {"ignore_missing_keys": True}
    lab = Imagelab(filepaths=image_paths, storage_opts=options, verbose=False)

    assert lab.issues.index.tolist() == image_paths[:1]
    assert options == {"ignore_missing_keys": True}


def test_storage_options_can_be_reused(image_paths):
    options = {"ignore_missing_keys": True}
    first = Imagelab(filepaths=image_paths, storage_opts=options, verbose=False)
    second = Imagelab(filepaths=image_paths, storage_opts=options, verbose=False)

    assert first.issues.index.tolist() == image_paths[:1]
    assert second.issues.index.tolist() == image_paths[:1]
    second.find_issues({"grayscale": {}}, n_jobs=1, verbose=False)
    assert len(second.issues) == 1
    assert second.issue_summary["num_images"].tolist() == [0]


def test_existing_file_without_storage_options(image_paths):
    lab = Imagelab(filepaths=image_paths[:1], verbose=False)
    lab.find_issues({"grayscale": {}}, n_jobs=1, verbose=False)

    assert lab.issues.index.tolist() == image_paths[:1]
    assert lab.issue_summary["num_images"].tolist() == [0]


def test_all_missing_files_raise_value_error(image_paths):
    with pytest.raises(ValueError, match="No images found"):
        Imagelab(
            filepaths=image_paths[1:],
            storage_opts={"ignore_missing_keys": True},
            verbose=False,
        )
