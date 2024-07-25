import os
import sys
import hashlib
import requests
import tarfile
import zipfile
from tqdm import tqdm

DATASETS = {
    "banana-detection": (
        "http://d2l-data.s3-accelerate.amazonaws.com/banana-detection.zip",
        "5de26c8fce5ccdea9f91267273464dc968d20d72",
    ),
    "Hardhat": (
        "https://drive.usercontent.google.com/download?id=1cZ1z7NAkbeus6LtGtuzBiK7RnUfe5Eck&export=download&authuser=0&confirm=t&uuid=3177a133-62ca-4f36-bcea-310164b29d1d&at=APZUnTWLlgE4qxu6cy4Y55XE0zIZ:1719074001874",
        "",
    ),
}


def sizeof_fmt(num: int | float) -> str:
    for x in ["bytes", "KB", "MB", "GB"]:
        if num < 1024.0:
            return "%.1f %s" % (num, x)
        num /= 1024.0
    return "%.1f %s" % (num, "TB")


def download(url: str, name=None, folder: str = "./data", sha1_hash=None):
    """Download a file to folder and return the local filepath."""
    if not url.startswith("http"):
        # For back compatability
        name = url + ".zip"
        url, sha1_hash = DATASETS[url]
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split("/")[-1] if name is None else name)
    # Check if hit cache
    if os.path.exists(fname):
        if sha1_hash:
            sha1 = hashlib.sha1()
            with open(fname, "rb") as f:
                while True:
                    data = f.read(1048576)
                    if not data:
                        break
                    sha1.update(data)
            if sha1.hexdigest() == sha1_hash:
                return fname
        else:
            return fname
    # Download
    print(f"Downloading {fname} from {url}...")
    rs = requests.get(url, stream=True, verify=True)
    total_size = int(rs.headers.get("content-length", 0))
    print("From content-length:", sizeof_fmt(total_size))
    chunk_size = 1024 * 1024
    num_bars = int(total_size / chunk_size)
    with open(fname, mode="wb") as f:
        for data in tqdm(rs.iter_content(chunk_size), total=num_bars, unit="MB", file=sys.stdout):
            f.write(data)
    return fname


def download_extract(url: str, name: str = None, folder: str = "./data"):
    """Download and extract a zip/tar file."""
    fname = download(url, name, folder=folder)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if not os.path.isdir(data_dir):
        if ext == ".zip":
            fp = zipfile.ZipFile(fname, "r")
        elif ext in (".tar", ".gz"):
            fp = tarfile.open(fname, "r")
        else:
            assert False, "Only zip/tar files can be extracted."
        fp.extractall(base_dir)
    return data_dir
