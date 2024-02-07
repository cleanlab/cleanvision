import sys
from typing import Any, Union

from cleanvision.imagelab import Imagelab as _Imagelab

PYTHON_VERSION_INFO = sys.version_info


def get_version() -> Union[str, Any]:
    if sys.version_info.major >= 3 and sys.version_info.minor >= 8:
        import importlib.metadata

        return importlib.metadata.version("cleanvision")
    else:
        import importlib_metadata

        return importlib_metadata.version("cleanvision")


try:
    __version__ = get_version()
except Exception:
    pass

Imagelab = _Imagelab
