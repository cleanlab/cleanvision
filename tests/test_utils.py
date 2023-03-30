from PIL import Image

from cleanvision.utils.utils import get_filepaths


class TestUtils:
    def test_get_filepaths(self, tmp_path):
        img = Image.new("L", (100, 100))
        extensions = [".png", ".jpeg", ".gif", ".jpg"]
        for i in range(2):
            for ext in extensions:
                p = tmp_path / f"img_{i}{ext}"
                img.save(p)
        filepaths = get_filepaths(tmp_path)
        print(filepaths)
        assert len(filepaths) == len(extensions) * 2
