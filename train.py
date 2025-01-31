import dlib
import os
from hyperparams import upsample_limit

current_directory = os.getcwd()
model_file = os.path.join(current_directory, "model.svm")


def train():
    options = dlib.simple_object_detector_training_options()
    options.be_verbose = True
    options.epsilon = 0.001
    options.C = 10
    options.num_threads = 15
    options.upsample_limit = upsample_limit

    detector = dlib.train_simple_object_detector(
        os.path.join(current_directory, "gtsdb-train.xml"), model_file, options
    )

    print("Testing the trained detector...")
    test_results = dlib.test_simple_object_detector(
        os.path.join(current_directory, "gtsdb-test.xml"), model_file
    )

    print(f"Average precision: {test_results.average_precision}")
    print(f"Precision: {test_results.precision}")
    print(f"Recall: {test_results.recall}")


train()
