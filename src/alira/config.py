import os


class _Config:
    def __getitem__(self, key):
        return os.environ.get(key)


CONFIG = _Config()
