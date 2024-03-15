import io
import os
import urllib.request
from io import IOBase

import aiohttp


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
    def __init__(self, filepath: str, mode: str) -> None:
        self.filepath = filepath
        self.mode = mode
        self.file = None

    def __enter__(self) -> IOBase:
        assert not self.filepath.startswith("az://"), "Azure blob storage is not supported"
        if self.filepath.startswith("https://"):
            assert self.mode in ["r", "rb"], "Only read mode is supported for remote files"
            remote_data = urllib.request.urlopen(self.filepath)
            if "b" in self.mode:
                # Read the content into a BytesIO object for binary mode
                self.file = io.BytesIO(remote_data.read())
            else:
                # Decode the content and use StringIO for text mode (less common for torch.load)
                self.file = io.StringIO(remote_data.read().decode())
        else:
            # Create the subdirectories if they don't exist
            directory = os.path.dirname(self.filepath)
            os.makedirs(directory, exist_ok=True)
            self.file = open(self.filepath, self.mode)
            if "b" in self.mode:
                # Ensure the file is seekable; if not, read into a BytesIO object
                try:
                    self.file.seek(0)
                except io.UnsupportedOperation:
                    self.file.close()
                    with open(self.filepath, self.mode) as f:
                        self.file = io.BytesIO(f.read())
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        # Close the file if it's open
        if self.file is not None:
            self.file.close()
        # Propagate exceptions
        return False


async def read_single_async(filepath: str) -> bytes:
    if filepath.startswith("https://"):
        async with aiohttp.ClientSession() as session:
            async with session.get(filepath) as response:
                return await response.read()
    else:
        with open(filepath, "rb") as f:
            return f.read()


def copy_to_local_cache(src: str, dst: str) -> None:
    if not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
    if src.startswith("https://"):
        with urllib.request.urlopen(src) as response, open(dst, "wb") as out_file:
            data = response.read()  # Consider chunked reading for large files
            out_file.write(data)
    else:
        with open(src, "rb") as in_file, open(dst, "wb") as out_file:
            data = in_file.read()
            out_file.write(data)
