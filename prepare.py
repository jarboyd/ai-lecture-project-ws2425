import os
import cv2
from collections import defaultdict
from random import shuffle
import xml.etree.cElementTree as ET
import math
from hyperparams import convert, shuffle_files, perf_take, train_split, upsample_limit

current_directory = os.getcwd()
dataset_path = os.path.join(current_directory, "FullIJCNN2013")

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

el_dataset = ET.Element("dataset")
el_images = ET.SubElement(el_dataset, "images")

files = os.listdir(dataset_path)
if shuffle_files:
    shuffle(files)
files = files[: int(len(files) * perf_take)]
files_train = files[: int(len(files) * train_split)]
files_test = files[int(len(files) * train_split) :]

min_box_area = math.pow(80, 2) * math.pow(2, -2 * upsample_limit)


def write_xml(files, filename):
    num_images = 0
    num_boxes = 0
    for file in files:
        if not file.endswith(".jpg"):
            continue
        boxes = ground_truths[file.replace(".jpg", ".ppm")]
        suitable_boxes = [
            (left, top, right, bottom, label)
            for left, top, right, bottom, label in boxes
            if (right - left) * (bottom - top) >= min_box_area
        ]
        if len(boxes) > 0 and len(suitable_boxes) == 0:
            continue
        num_images += 1
        el_image = ET.SubElement(
            el_images, "image", file=os.path.join(dataset_path, file)
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


num_images, num_boxes = write_xml(
    files_train, os.path.join(current_directory, "gtsdb-train.xml")
)
print(f"Found {num_images} suitable images with {num_boxes} boxes for training")
num_images, num_boxes = write_xml(
    files_test, os.path.join(current_directory, "gtsdb-test.xml")
)
print(f"Found {num_images} suitable images with {num_boxes} boxes for training")
