import os

# dataset metadata
categories = {
    "prohibitory": [
        0,
        1,
        2,
        3,
        4,
        5,
        7,
        8,
        9,
        10,
        15,
        16,
        12,
    ],  # prohibitory, circular, double edge + priority road
    "mandatory": [33, 34, 35, 36, 37, 38, 39, 40],  # mandatory, circular, single edge
    "danger": [
        11,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
    ],  # danger, upward triangle, double edge
    "restriction_ends": [
        6,
        32,
        41,
        42,
    ],  # other/restriction ends, circular, single edge
    "give_way": [13],  # other/give way, rectangular, single edge
    "stop": [14],  # other/stop, octagonal, single edge
    "no_entry": [17],  # other/no entry, single edge
}

# preprocessing
# convert images from .ppm to .jpg; only needs to be done once
convert = True
# shuffle images before splitting
shuffle_files = True
# use partial dataset for training; must be in range (0,1)
train_split = 0.8

# training
# upsample images, heavy RAM usage; must be an integer
upsample_limit = 2

# performance
# use only partial dataset to reduce RAM usage; must be in range (0,1)
perf_take = 1
# test the dataset immediately after training; disable to save time
test_after_training = True

# miscellaneous
# several directories used throughout the project
__current_directory = os.getcwd()
dataset_path = os.path.join(__current_directory, "FullIJCNN2013")
__resources_directory = os.path.join(__current_directory, "resources")
splits_directory = os.path.join(__resources_directory, "splits")
models_directory = os.path.join(__resources_directory, "models")
