import os
import cv2
import numpy as np
import dlib
import time
from random import shuffle
from hyperparams import upsample_limit

models_directory = os.path.join(os.getcwd(), "resources", "models")
# Comprehension which loads the detectors and holds them in a dictionary with
# their respective name
detectors = {
    model[6:-4]: dlib.simple_object_detector(os.path.join(models_directory, model))
    for model in os.listdir(models_directory)
}

images_folder = os.path.join(os.getcwd(), "FullIJCNN2013")
# Confidence threshold to display detections
confidence_threshold = 0.5

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
font_thickness = 2

files = os.listdir(images_folder)
shuffle(files)
for file in files:
    # Process only .jpg files
    if not file.endswith(".jpg"):
        continue

    img = cv2.imread(os.path.join(images_folder, file))

    # Track how long it takes to run the prediction
    start_time = time.time()
    # Detections contain the bounding boxes, corresponding confidences in the
    # detections and which detector made the detection
    boxes, confidences, detector_idxs = dlib.simple_object_detector.run_multiple(
        list(detectors.values()),
        img,
        upsample_num_times=upsample_limit,
        adjust_threshold=0.0,
    )
    time_taken = time.time() - start_time

    custom_annotation = f"time_taken: {time_taken:.3f}"
    annotation_position = (img.shape[1] - 250, 30)
    cv2.putText(
        img,
        custom_annotation,
        annotation_position,
        font,
        font_scale,
        (0, 0, 255),
        font_thickness,
    )

    for i in range(len(boxes)):
        # Skip if confidence is below threshold
        if confidences[i] < confidence_threshold:
            continue

        box = boxes[i]
        label = (
            f"Detector {list(detectors.keys())[detector_idxs[i]]}: {confidences[i]:.2f}"
        )

        # Draw bounding box
        cv2.rectangle(
            img, (box.left(), box.top()), (box.right(), box.bottom()), (0, 0, 255), 2
        )

        # Display label and confidence below the box
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = box.left()
        text_y = box.bottom() + text_size[1] + 5

        cv2.putText(
            img, label, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness
        )

    # Display the image with annotations
    cv2.imshow("Annotated Image", img)

    # Wait for a key press, press 'n' to move to the next image, the attentive code
    # reader may see that no matter what key you press, it will continue with the next image
    key = cv2.waitKey(0) & 0xFF
    if key == ord("n"):
        continue
    # Quit if 'q' is pressed
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
