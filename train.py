import dlib
import os
from hyperparams import (
    models_directory,
    splits_directory,
    upsample_limit,
    categories,
    test_after_training,
)


if not os.path.isdir(models_directory):
    os.mkdir(models_directory)


def train(category):
    options = dlib.simple_object_detector_training_options()
    options.be_verbose = True
    options.epsilon = 0.001
    options.C = 10
    options.num_threads = 15
    options.upsample_limit = upsample_limit

    model_file = os.path.join(models_directory, f"model_{category}.svm")

    print(f"Training detector for category '{category}'")
    dlib.train_simple_object_detector(
        os.path.join(splits_directory, f"gtsdb-train_{category}.xml"),
        model_file,
        options,
    )

    if test_after_training:
        print("Testing the trained detector...")
        test_results = dlib.test_simple_object_detector(
            os.path.join(splits_directory, f"gtsdb-test_{category}.xml"), model_file
        )

        print(f"Average precision: {test_results.average_precision}")
        print(f"Precision: {test_results.precision}")
        print(f"Recall: {test_results.recall}")


for category in categories.keys():
    train(category)
