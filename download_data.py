"""Download and verify the unchanged official EPFL Project 1 CSV dataset.

Uses only the Python standard library. Both the archive and extracted CSVs are
verified against SHA-256 digests of the official course download. Existing CSVs
are reused only after all four digests match. Downloads and file replacements
are atomic, and extraction accepts only the four expected dataset members.
"""

import argparse
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
import zipfile

DATA_URL = (
    "https://raw.githubusercontent.com/epfml/ML_course/main/"
    "projects/project1/data/dataset_to_release.zip"
)
ARCHIVE_SHA256 = "45812b9333df561c72bed861ba5ec382ddc2b68defd64ddda64118074d85a0fd"
FILE_SHA256 = {
    "x_train.csv": "bfa447edece27d45b54f8f1486f6351a381885968d71c9eb1574fd1dbf0fa14a",
    "y_train.csv": "bc974ff01bd82827e05780d106674ee680c918fc54225384cbf4f7b279422ec1",
    "x_test.csv": "28a7f5f94436a33b5dceb5dd15fd72563f31ec898124e382c123d3827203503c",
    "sample_submission.csv": "62357ed950e0a7eb238745656fdefaecf451bb79e1775918433472974566a9e3",
}


def sha256(path):
    """Compute a streaming SHA-256 without retaining a full CSV in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive(archive_path):
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=archive_path.parent, prefix="dataset-download-", suffix=".zip", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        request = urllib.request.Request(
            DATA_URL, headers={"User-Agent": "EPFL-project1"}
        )
        digest = hashlib.sha256()
        downloaded = 0
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary_path.open("wb") as output:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
        if digest.hexdigest() != ARCHIVE_SHA256:
            raise ValueError(
                "Official archive checksum differs from this project's recorded dataset; "
                "the download was not installed."
            )
        os.replace(temporary_path, archive_path)
        print(
            f"Downloaded and verified {downloaded:,} bytes from the course repository."
        )
    finally:
        temporary_path.unlink(missing_ok=True)


def _extract_archive(archive_path, data_dir):
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP members are not allowed")
        for member in members:
            path = PurePosixPath(member.filename)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.filename
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        expected = {"dataset/" + name for name in FILE_SHA256}
        if not expected.issubset(names):
            raise ValueError("The archive does not contain all four expected CSV files")
        # Stage every CSV and verify all contents before replacing existing files.
        with tempfile.TemporaryDirectory(
            dir=data_dir.parent, prefix="dataset-extract-"
        ) as staging:
            staging = Path(staging)
            for name, expected_digest in FILE_SHA256.items():
                path = staging / name
                with archive.open("dataset/" + name) as source, path.open(
                    "wb"
                ) as target:
                    shutil.copyfileobj(source, target, length=1 << 20)
                if sha256(path) != expected_digest:
                    raise ValueError(f"Extracted file checksum mismatch: {name}")
            data_dir.mkdir(parents=True, exist_ok=True)
            for name in FILE_SHA256:
                os.replace(staging / name, data_dir / name)


def download_data(data_dir="data/dataset"):
    """Ensure the four official CSVs exist and match their recorded SHA-256s."""
    data_dir = Path(data_dir).resolve()
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    complete = all(
        (data_dir / name).is_file() and sha256(data_dir / name) == expected
        for name, expected in FILE_SHA256.items()
    )
    if complete:
        print(f"Verified all four official CSV files in {data_dir}.")
        return data_dir
    archive_path = data_dir.parent / "dataset_to_release.zip"
    if not archive_path.is_file() or sha256(archive_path) != ARCHIVE_SHA256:
        _download_archive(archive_path)
    else:
        print("Reusing the verified official dataset archive.")
    _extract_archive(archive_path, data_dir)
    print(f"Extracted and verified all four official CSV files in {data_dir}.")
    return data_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dataset"))
    args = parser.parse_args()
    download_data(args.data_dir)
