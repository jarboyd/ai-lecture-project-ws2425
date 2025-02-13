import os
import cv2
from collections import defaultdict
from random import shuffle
import xml.etree.cElementTree as ET
import math
from hyperparams import (
    categories,
    convert,
    shuffle_files,
    perf_take,
    train_split,
    upsample_limit,
    dataset_path,
    splits_directory,
)

if convert:
    for file in os.listdir(dataset_path):
        if not file.endswith(".ppm"):
            continue
        el_image = cv2.imread(os.path.join(dataset_path, file))
        cv2.imwrite(os.path.join(dataset_path, file.replace(".ppm", ".jpg")), el_image)

ground_truths = defaultdict(lambda: [])
with open(os.path.join(dataset_path, "gt.txt")) as f:
    lines = f.readlines()
    for line in lines:
        name, x1, y1, x2, y2, label = line.split(";")
        ground_truths[name].append((int(x1), int(y1), int(x2), int(y2), int(label)))

files = [file for file in os.listdir(dataset_path) if file.endswith(".ppm")]

label_groups_files = {
    key: [
        file
        for file in files
        if any([label in labels for _, _, _, _, label in ground_truths[file]])
    ]
    for key, labels in categories.items()
}

min_box_area = math.pow(80, 2) * math.pow(2, -2 * upsample_limit)


def write_xml(files, filename, labels=[], skip_checks=False):
    num_images = 0
    num_boxes = 0
    el_dataset = ET.Element("dataset")
    el_images = ET.SubElement(el_dataset, "images")
    for file in files:
        boxes = [
            (left, top, right, bottom, label)
            for left, top, right, bottom, label in ground_truths[file]
            if skip_checks or label in labels
        ]
        suitable_boxes = [
            (left, top, right, bottom, label)
            for left, top, right, bottom, label in boxes
            if skip_checks or (right - left) * (bottom - top) >= min_box_area
        ]
        if not skip_checks and len(boxes) > 0 and len(suitable_boxes) == 0:
            continue
        num_images += 1
        el_image = ET.SubElement(
            el_images,
            "image",
            file=os.path.join(dataset_path, file.replace(".ppm", ".jpg")),
        )
        for left, top, right, bottom, label in suitable_boxes:
            num_boxes += 1
            width = right - left
            height = bottom - top
            if width * height < min_box_area:
                continue
            ET.SubElement(
                el_image,
                "box",
                top=str(top),
                left=str(left),
                width=str(width),
                height=str(height),
            )
            ET.SubElement(el_image, "label").text = str(label)
    tree = ET.ElementTree(el_dataset)
    tree.write(filename)
    return num_images, num_boxes


if not os.path.isdir(splits_directory):
    os.makedirs(splits_directory)
for category, category_files in label_groups_files.items():
    if shuffle_files:
        shuffle(category_files)
    category_files = category_files[: int(len(category_files) * perf_take)]
    files_train = category_files[: int(len(category_files) * train_split)]
    files_test = category_files[int(len(category_files) * train_split) :]
    print(f"Generating file lists for category '{category}':")
    num_images, num_boxes = write_xml(
        files_train,
        os.path.join(splits_directory, f"gtsdb-train_{category}.xml"),
        categories[category],
    )
    print(
        f"\tFound {num_images} out of {len(files_train)} suitable images with {num_boxes} boxes for training"
    )
    num_images, num_boxes = write_xml(
        files_test,
        os.path.join(splits_directory, f"gtsdb-test_{category}.xml"),
        categories[category],
    )
    print(
        f"\tFound {num_images} out of {len(files_test)} suitable images with {num_boxes} boxes for testing\n"
    )

print("Generating full file list for validation:")
num_images, num_boxes = write_xml(
    files, os.path.join(splits_directory, f"gtsdb-full.xml"), skip_checks=True
)
print(
    f"\tFound {num_images} out of {len(files)} suitable images with {num_boxes} boxes for testing\n"
)
