"""Download the Kaggle water-bodies dataset to data/ using kagglehub."""

import os, shutil
from pathlib import Path

REPO = "franciscoescobar/satellite-images-of-water-bodies"
DEST = Path("data")

def download():
    import kagglehub
    path = kagglehub.dataset_download(REPO)
    print(f"Downloaded to {path}")

    os.makedirs(DEST, exist_ok=True)

    for f in Path(path).iterdir():
        shutil.move(str(f), DEST / f.name)

    print(f"Moved to {DEST.resolve()}/")
    print(f"Contents: {os.listdir(DEST)}")


if __name__ == "__main__":
    download()
