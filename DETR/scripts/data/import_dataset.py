from pathlib import Path
import shutil

import kagglehub

DATASET_HANDLE = "maitam/vietnamese-traffic-signs"
DATASET_ROOT = Path(__file__).resolve().parents[3] / "dataset"

download_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
if not download_path.is_dir():
    raise NotADirectoryError(f"KaggleHub returned a missing dataset directory: {download_path}")

if download_path.resolve() == DATASET_ROOT.resolve():
    raise RuntimeError("KaggleHub download path unexpectedly matches the dataset destination.")

DATASET_ROOT.mkdir(parents=True, exist_ok=True)
copied_files = 0
existing_files = 0

for source in download_path.rglob("*"):
    if not source.is_file():
        continue

    destination = DATASET_ROOT / source.relative_to(download_path)
    if destination.exists():
        if not destination.is_file():
            raise FileExistsError(f"Expected a file at dataset destination: {destination}")
        if destination.read_bytes() != source.read_bytes():
            raise FileExistsError(
                f"Refusing to overwrite a different existing file: {destination}"
            )
        existing_files += 1
        continue

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied_files += 1

print(f"Dataset available at: {DATASET_ROOT}")
print(f"Copied {copied_files} file(s); kept {existing_files} identical existing file(s).")