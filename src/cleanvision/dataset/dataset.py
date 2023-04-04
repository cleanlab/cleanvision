from PIL import Image

from cleanvision.utils.utils import get_filepaths


class Dataset:
    """This class is used for managing different kinds of data formats provided by user"""

    def __init__(self):
        self.index = None

    def _set_index(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        raise NotImplementedError

    def get_name(self, index):
        raise NotImplementedError


class HFDataset(Dataset):
    def __init__(self, hf_dataset, image_key):
        super().__init__()
        self._data = hf_dataset
        self._image_key = image_key
        self._set_index()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item][self._image_key]

    def __next__(self):
        if self._idx >= len(self._data):
            raise StopIteration
        # todo : catch reading image errors
        index, image = self._idx, self._data[self._idx][self._image_key]
        self._idx += 1
        return index, image

    def _set_index(self):
        self.index = [i for i in range(len(self._data))]

    def get_name(self, index):
        return f"idx: {index}"


class FolderDataset(Dataset):
    def __init__(self, data_folder):
        super().__init__()
        self._filepaths = get_filepaths(data_folder)
        self._set_index()

    def __len__(self):
        return len(self._filepaths)

    def __getitem__(self, item):
        return Image.open(item)

    def __next__(self):
        if self._idx >= len(self._filepaths):
            raise StopIteration
        index, image = self._filepaths[self._idx], Image.open(
            self._filepaths[self._idx]
        )
        self._idx += 1
        return index, image

    def _set_index(self):
        self.index = self._filepaths.copy()

    def get_name(self, index):
        return index.split("/")[-1]


class FilePathDataset(Dataset):
    def __init__(self, filepaths):
        super().__init__()
        self._filepaths = filepaths
        self._set_index()

    def __len__(self):
        return len(self._filepaths)

    def __getitem__(self, item):
        return Image.open(self._filepaths[item])

    def __next__(self):
        if self._idx >= len(self._filepaths):
            raise StopIteration
        index, image = self._filepaths[self._idx], Image.open(
            self._filepaths[self._idx]
        )
        self._idx += 1
        return index, image

    def _set_index(self):
        self.index = self._filepaths.copy()


# todo
class TorchDataset(Dataset):
    def __init__(self, torch_dataset):
        super().__init__()
        self._data = torch_dataset
        # todo: catch errors
        for i, obj in enumerate(torch_dataset[0]):
            if isinstance(obj, Image.Image):
                self._image_idx = i
        self._set_index()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item][self._image_idx]

    def __next__(self):
        if self._idx >= len(self._data):
            raise StopIteration
        # todo : catch reading image errors
        index, image = self._idx, self._data[self._idx][self._image_idx]
        self._idx += 1
        return index, image

    def get_name(self, index):
        return f"idx: {index}"

    def _set_index(self):
        self.index = [i for i in range(len(self._data))]
