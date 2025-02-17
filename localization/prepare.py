import os
import cv2
from collections import defaultdict
from random import shuffle
import xml.etree.cElementTree as ET
import math

# These imports serve as shared variables
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

# If ther convert flag is set
if convert:
    for file in os.listdir(dataset_path):
        if not file.endswith(".ppm"):
            continue
        # Dlib cannot read .ppm files, so convert them to .jpg
        el_image = cv2.imread(os.path.join(dataset_path, file))
        cv2.imwrite(os.path.join(dataset_path, file.replace(".ppm", ".jpg")), el_image)

ground_truths = defaultdict(lambda: [])
# The gt.txt file contains labels for all bounding boxes (and sign types) in the images
with open(os.path.join(dataset_path, "gt.txt")) as f:
    lines = f.readlines()
    for line in lines:
        name, x1, y1, x2, y2, label = line.split(";")
        # ground_truths is a dictionary containing lists of labels
        # Each label has the shape (left, top, right, bottom, label)
        # Where left,top,right,bottom are pixels that define the bounding box
        # And label is in [0,42], each number corresponding to a certain german street sign
        ground_truths[name].append((int(x1), int(y1), int(x2), int(y2), int(label)))

files = [file for file in os.listdir(dataset_path) if file.endswith(".ppm")]

# This comprehension groups together all files by their category
# Where a category is a list of labels whose corresponding signs are of similar shape
# which helps the model training
label_groups_files = {
    key: [
        # File name
        file
        for file in files
        # If any of the labels in the image are in the allowed
        # labels for the current category
        if any([label in labels for _, _, _, _, label in ground_truths[file]])
    ]
    for key, labels in categories.items()
}

# The minimum area for a bounding box depends on the upsampling count set
# The higher the upsampling count the lower the minimum size
# Without upsampling, the size is 1600 pixels in area (80*80)
# And upsampling squares the pixel count, cutting the minimal area
# by 4 respectively
min_box_area = math.pow(80, 2) * math.pow(2, -2 * upsample_limit)


def write_xml(files, filename, labels=[], skip_checks=False):
    num_images = 0
    num_boxes = 0
    # The root for the dataset
    el_dataset = ET.Element("dataset")
    # Container for a list of images
    el_images = ET.SubElement(el_dataset, "images")
    for file in files:
        # Only boxes with an allowed label are selected
        boxes = [
            (left, top, right, bottom, label)
            for left, top, right, bottom, label in ground_truths[file]
            if skip_checks or label in labels
        ]
        # Only boxes with at least the minimum required area are selected
        suitable_boxes = [
            (left, top, right, bottom, label)
            for left, top, right, bottom, label in boxes
            if skip_checks or (right - left) * (bottom - top) >= min_box_area
        ]
        # If there were street signs to be found in the image, but they are
        # not suitable,  discard the image to not learn to falsely predict
        # "no signs found"
        if not skip_checks and len(boxes) > 0 and len(suitable_boxes) == 0:
            continue
        num_images += 1
        # The final training dataset format is in XML and is constructed here
        # Each image starts with an image tag containing the file name as a base
        # And is appended to the images container
        el_image = ET.SubElement(
            el_images,
            "image",
            file=os.path.join(dataset_path, file.replace(".ppm", ".jpg")),
        )
        for left, top, right, bottom, label in suitable_boxes:
            num_boxes += 1
            # Redundant check, but I'll leave it in just in case it wasn't and
            # I included it to fix something
            width = right - left
            height = bottom - top
            if width * height < min_box_area:
                continue
            # Each box is appended to the image node, which serves as a container
            ET.SubElement(
                el_image,
                "box",
                top=str(top),
                left=str(left),
                width=str(width),
                height=str(height),
            )
            # Unused, but expected in the XML format, also, now that I
            # read it again, it overwrites the label for each box written
            # to the image
            ET.SubElement(el_image, "label").text = str(label)
    # Construct and write the XML
    tree = ET.ElementTree(el_dataset)
    tree.write(filename)
    return num_images, num_boxes


if not os.path.isdir(splits_directory):
    os.makedirs(splits_directory)
# category will contain the category name
# category_files will contain all image file names with labels for said category
for category, category_files in label_groups_files.items():
    # If shuffle flag is set, shuffle files in place
    if shuffle_files:
        shuffle(category_files)
    # To reduce RAM usage, perf_take can reduce the total amount of files used
    # for training
    category_files = category_files[: int(len(category_files) * perf_take)]
    # Full dataset is split into train and test splits
    files_train = category_files[: int(len(category_files) * train_split)]
    files_test = category_files[int(len(category_files) * train_split) :]
    # Ensure test has at least one image, this will still break if there is
    # only one image in a category, but let's just pretend that won't happen
    if len(files_test) == 0:
        files_train = files_train[:-1]
        files_test = files_train[-1:]
    print(f"Generating file lists for category '{category}':")
    # Generate the XML file for training data ...
    num_images, num_boxes = write_xml(
        files_train,
        os.path.join(splits_directory, f"gtsdb-train_{category}.xml"),
        categories[category],
    )
    print(
        f"\tFound {num_images} out of {len(files_train)} suitable images with {num_boxes} boxes for training"
    )
    # ... and test data respectively
    num_images, num_boxes = write_xml(
        files_test,
        os.path.join(splits_directory, f"gtsdb-test_{category}.xml"),
        categories[category],
    )
    print(
        f"\tFound {num_images} out of {len(files_test)} suitable images with {num_boxes} boxes for testing\n"
    )

print("Generating full file list for validation:")
# Not a category, but generate XML for all images just to have it for validation
num_images, num_boxes = write_xml(
    files, os.path.join(splits_directory, f"gtsdb-full.xml"), skip_checks=True
)
print(
    f"\tFound {num_images} out of {len(files)} suitable images with {num_boxes} boxes for testing\n"
)
