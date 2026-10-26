from collections import Counter
import json
import math
from pathlib import Path
import random
import shutil
import tempfile

from PIL import Image


DATASET_ROOT = Path(__file__).resolve().parents[3] / "dataset"
SOURCE_ROOT = DATASET_ROOT / "archive"
OUTPUT_ROOT = DATASET_ROOT / "model_ready"
VALIDATION_FRACTION = 0.1
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_split(path: Path) -> list[str]:
    filenames = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(filenames) != len(set(filenames)):
        raise ValueError(f"Duplicate image filenames in split list: {path}")
    for filename in filenames:
        if Path(filename).name != filename:
            raise ValueError(f"Expected a filename, not a path, in {path}: {filename}")
    return filenames


def load_categories() -> list[dict[str, object]]:
    names = [
        line.strip()
        for line in (SOURCE_ROOT / "classes_en.txt")
        .read_text(encoding="utf-8-sig")
        .splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"No class names found in {SOURCE_ROOT / 'classes_en.txt'}")
    return [{"id": class_id + 1, "name": name} for class_id, name in enumerate(names)]


def split_training_files(filenames: list[str], class_count: int) -> tuple[list[str], list[str]]:
    image_classes = {}
    class_image_counts: Counter[int] = Counter()
    label_root = SOURCE_ROOT / "labels"

    for filename in filenames:
        label_path = label_root / Path(filename).with_suffix(".txt")
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label file: {label_path}")
        classes = set()
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            fields = line.split()
            if not fields:
                continue
            try:
                class_id = int(fields[0])
            except ValueError as exc:
                raise ValueError(
                    f"{label_path}:{line_number}: invalid class ID"
                ) from exc
            if not 0 <= class_id < class_count:
                raise ValueError(
                    f"{label_path}:{line_number}: class ID {class_id} is out of range"
                )
            classes.add(class_id)
        image_classes[filename] = classes
        class_image_counts.update(classes)

    validation_count = max(1, round(len(filenames) * VALIDATION_FRACTION))
    validation_targets = {
        class_id: min(
            max(1, round(image_count * VALIDATION_FRACTION)),
            image_count - 1,
        )
        for class_id, image_count in class_image_counts.items()
        if image_count > 1
    }
    rng = random.Random(RANDOM_SEED)
    remaining = set(filenames)
    validation = []
    validation_class_counts: Counter[int] = Counter()

    while len(validation) < validation_count:
        candidates = []
        best_score = 0.0
        for filename in sorted(remaining):
            if any(
                validation_class_counts[class_id] + 1
                >= class_image_counts[class_id]
                for class_id in image_classes[filename]
            ):
                continue
            score = sum(
                1 / class_image_counts[class_id]
                for class_id in image_classes[filename]
                if validation_class_counts[class_id] < validation_targets.get(
                    class_id, 0
                )
            )
            if score > best_score:
                candidates = [filename]
                best_score = score
            elif score == best_score and score > 0:
                candidates.append(filename)

        if not candidates:
            break

        selected = rng.choice(candidates)
        validation.append(selected)
        remaining.remove(selected)
        validation_class_counts.update(image_classes[selected])

    if len(validation) < validation_count:
        while len(validation) < validation_count:
            candidates = sorted(
                filename
                for filename in remaining
                if all(
                    validation_class_counts[class_id] + 1
                    < class_image_counts[class_id]
                    for class_id in image_classes[filename]
                )
            )
            if not candidates:
                raise ValueError(
                    "Cannot create the requested validation split while keeping "
                    "every training class represented."
                )
            selected = rng.choice(candidates)
            validation.append(selected)
            remaining.remove(selected)
            validation_class_counts.update(image_classes[selected])

    validation_set = set(validation)
    train = sorted(set(filenames) - validation_set)
    if not train:
        raise ValueError("The validation split would leave no training images.")
    return train, sorted(validation_set)


def validate_annotation(
    label_path: Path, class_count: int
) -> tuple[list[tuple[int, float, float, float, float, bool]], list[str]]:
    boxes = []
    errors = []

    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            errors.append(
                f"{label_path}:{line_number}: expected 5 fields, found {len(fields)}"
            )
            continue

        try:
            class_id = int(fields[0])
            cx, cy, box_width, box_height = map(float, fields[1:])
        except ValueError as exc:
            errors.append(f"{label_path}:{line_number}: invalid value: {exc}")
            continue

        if not 0 <= class_id < class_count:
            errors.append(
                f"{label_path}:{line_number}: class ID {class_id} is out of range"
            )
            continue
        if not all(math.isfinite(value) for value in (cx, cy, box_width, box_height)):
            errors.append(f"{label_path}:{line_number}: non-finite box coordinate")
            continue
        if box_width <= 0 or box_height <= 0:
            errors.append(f"{label_path}:{line_number}: box size must be positive")
            continue

        raw_left, raw_top = cx - box_width / 2, cy - box_height / 2
        raw_right, raw_bottom = cx + box_width / 2, cy + box_height / 2
        left = max(0.0, raw_left)
        top = max(0.0, raw_top)
        right = min(1.0, raw_right)
        bottom = min(1.0, raw_bottom)
        if right <= left or bottom <= top:
            errors.append(
                f"{label_path}:{line_number}: box does not intersect the image"
            )
            continue

        clipped = (
            left != raw_left
            or top != raw_top
            or right != raw_right
            or bottom != raw_bottom
        )
        boxes.append((class_id, left, top, right, bottom, clipped))

    return boxes, errors


def prepare_split(
    split: str, filenames: list[str], categories: list[dict[str, object]]
) -> tuple[
    dict[str, object],
    list[tuple[Path, str]],
    Counter[int],
    int,
    list[str],
]:
    images = []
    annotations = []
    files_to_copy = []
    class_counts: Counter[int] = Counter()
    errors = []
    clipped_boxes = 0
    image_root = SOURCE_ROOT / "images"
    label_root = SOURCE_ROOT / "labels"

    for image_id, filename in enumerate(filenames, start=1):
        image_path = image_root / filename
        label_path = label_root / Path(filename).with_suffix(".txt")
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            errors.append(f"Unsupported image extension: {image_path}")
            continue
        if not image_path.is_file():
            errors.append(f"Missing image: {image_path}")
            continue
        if not label_path.is_file():
            errors.append(f"Missing label file: {label_path}")
            continue

        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
        except (OSError, ValueError) as exc:
            errors.append(f"Unreadable image {image_path}: {exc}")
            continue

        boxes, annotation_errors = validate_annotation(
            label_path, len(categories)
        )
        errors.extend(annotation_errors)
        images.append(
            {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
            }
        )
        files_to_copy.append((image_path, filename))

        for class_id, left, top, right, bottom, clipped in boxes:
            x = left * width
            y = top * height
            pixel_width = (right - left) * width
            pixel_height = (bottom - top) * height
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [x, y, pixel_width, pixel_height],
                    "area": pixel_width * pixel_height,
                    "iscrowd": 0,
                }
            )
            class_counts[class_id] += 1
            clipped_boxes += int(clipped)

    coco = {
        "info": {
            "description": "Vietnamese Traffic Signs detection dataset",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return coco, files_to_copy, class_counts, clipped_boxes, errors


def write_yolo_labels(
    coco: dict[str, object], labels_directory: Path
) -> None:
    images = {image["id"]: image for image in coco["images"]}
    labels_by_image = {image_id: [] for image_id in images}

    for annotation in coco["annotations"]:
        image = images[annotation["image_id"]]
        x, y, width, height = annotation["bbox"]
        center_x = (x + width / 2) / image["width"]
        center_y = (y + height / 2) / image["height"]
        normalized_width = width / image["width"]
        normalized_height = height / image["height"]
        yolo_class_id = annotation["category_id"] - 1
        labels_by_image[image["id"]].append(
            f"{yolo_class_id} {center_x:.10g} {center_y:.10g} "
            f"{normalized_width:.10g} {normalized_height:.10g}"
        )

    for image_id, image in images.items():
        (labels_directory / Path(image["file_name"]).with_suffix(".txt")).write_text(
            "\n".join(labels_by_image[image_id])
            + ("\n" if labels_by_image[image_id] else ""),
            encoding="utf-8",
        )


def main() -> None:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Kaggle dataset archive not found: {SOURCE_ROOT}")
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"{OUTPUT_ROOT} already exists. Move or rename it before preprocessing."
        )

    categories = load_categories()
    train_filenames = read_split(SOURCE_ROOT / "split_dataset" / "train_files.txt")
    test_filenames = read_split(SOURCE_ROOT / "split_dataset" / "test_files.txt")
    overlap = set(train_filenames) & set(test_filenames)
    if overlap:
        raise ValueError(f"Train/test lists overlap; example: {sorted(overlap)[0]}")
    if not train_filenames or not test_filenames:
        raise ValueError("Both train and test split lists must contain images.")

    final_train_filenames, validation_filenames = split_training_files(
        train_filenames, len(categories)
    )

    split_filenames = {
        "train": final_train_filenames,
        "validation": validation_filenames,
        "test": sorted(test_filenames),
    }
    prepared = {}
    errors = []
    for split, filenames in split_filenames.items():
        prepared[split] = prepare_split(split, filenames, categories)
        errors.extend(prepared[split][4])

    if errors:
        print("\n".join(errors[:50]))
        raise RuntimeError(
            f"Found {len(errors)} dataset issue(s); no output was written."
        )

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".model-ready-", dir=DATASET_ROOT
    ) as temporary_directory:
        staging_root = Path(temporary_directory) / "output"
        (staging_root / "images").mkdir(parents=True)
        (staging_root / "annotations").mkdir()
        (staging_root / "labels").mkdir()

        for split, (_, files_to_copy, _, _, _) in prepared.items():
            split_images = staging_root / "images" / split
            split_images.mkdir()
            for source, filename in files_to_copy:
                shutil.copy2(source, split_images / filename)

            coco = prepared[split][0]
            annotation_path = staging_root / "annotations" / f"instances_{split}.json"
            annotation_path.write_text(
                json.dumps(coco, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            split_labels = staging_root / "labels" / split
            split_labels.mkdir()
            write_yolo_labels(coco, split_labels)

        yolo_config = {
            "path": ".",
            "train": "images/train",
            "val": "images/validation",
            "test": "images/test",
            "nc": len(categories),
            "names": [category["name"] for category in categories],
        }
        (staging_root / "data.yaml").write_text(
            json.dumps(yolo_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        staging_root.replace(OUTPUT_ROOT)

    all_listed = set(train_filenames) | set(test_filenames)
    source_images = {
        path.name for path in (SOURCE_ROOT / "images").iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    unlisted_count = len(source_images - all_listed)

    for split, filenames in split_filenames.items():
        coco, _, class_counts, clipped_boxes, _ = prepared[split]
        print(
            f"{split}: {len(filenames)} images, "
            f"{len(coco['annotations'])} boxes, "
            f"{len(class_counts)} classes represented, "
            f"{clipped_boxes} boxes clipped to image bounds"
        )
    print(f"Unlisted source images skipped: {unlisted_count}")
    print(f"COCO and YOLO datasets ready at: {OUTPUT_ROOT}")
    print("COCO category IDs are 1-based; YOLO class IDs remain 0-based.")


if __name__ == "__main__":
    main()
