try:  # not supported on Python 3.7
    import importlib.metadata
    __version__ = importlib.metadata.version("cleanvision")
except Exception:
    pass
