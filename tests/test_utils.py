from PIL import Image

from cleanvision.utils.utils import get_filepaths


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
