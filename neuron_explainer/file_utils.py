import http
import os
import urllib.request
from io import IOBase


def file_exists(filepath: str) -> bool:
    if filepath.startswith("https://"):
        try:
            urllib.request.urlopen(filepath)
            return True
        except urllib.error.HTTPError:
            return False
    else:
        # It's a local file.
        return os.path.exists(filepath)


class CustomFileHandler:
    def __init__(self, filepath, mode) -> None:
        self.filepath = filepath
        self.mode = mode
        self.file = None

    def __enter__(self) -> IOBase | http.client.HTTPResponse:
        assert not self.filepath.startswith("az://"), "Azure blob storage is not supported"
        if self.filepath.startswith("https://"):
            assert self.mode in ["r", "rb"], "Only read mode is supported for remote files"
            self.file = urllib.request.urlopen(self.filepath)
        else:
            self.file = open(self.filepath, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Close the file if it's open
        if self.file is not None:
            self.file.close()
        # Propagate exceptions
        return False


def read_single_file(filepath: str) -> bytes:
    with CustomFileHandler(filepath, "rb") as f:
        return f.read()
