try:  # import only works for python>=3.8
    import importlib.metadata

    __version__ = importlib.metadata.version("cleanvision")
except Exception:
    pass
