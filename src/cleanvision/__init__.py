try:
    import importlib.metadata

    __version__ = importlib.metadata.version("cleanvision")
except Exception:
    pass
