import os


class _Config:
    def __getitem__(self, key):
        return os.environ[key]


CONFIG = _Config()
