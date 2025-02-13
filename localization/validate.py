import os
import cv2
import numpy as np
import dlib
import time
from random import shuffle

models_directory = os.path.join(os.getcwd(), "resources", "models")
detectors = {
    model[6:-4]: dlib.simple_object_detector(os.path.join(models_directory, model))
    for model in os.listdir(models_directory)
}

images_folder = os.path.join(os.getcwd(), "FullIJCNN2013")
confidence_threshold = 0.5  # Confidence threshold to display detections

# Adjusted text for readability
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.7  # Larger scale for better readability
font_thickness = 2  # Thicker text for better visibility

files = os.listdir(images_folder)
shuffle(files)
for file in files:
    if not file.endswith(".jpg"):
        continue

    img = cv2.imread(os.path.join(images_folder, file))

    # Assuming 'boxes', 'confidences', 'detector_idxs' are returned from your model
    start_time = time.time()
    boxes, confidences, detector_idxs = dlib.simple_object_detector.run_multiple(
        list(detectors.values()), img, upsample_num_times=1, adjust_threshold=0.0
    )
    time_taken = time.time() - start_time

    # Custom annotation text for the top-right corner
    custom_annotation = f"time_taken: {time_taken:.3f}"
    annotation_position = (img.shape[1] - 250, 30)  # Position near the top-right corner
    cv2.putText(img, custom_annotation, annotation_position, font, font_scale, (0, 0, 255), font_thickness)

    for i in range(len(boxes)):
        if confidences[i] < confidence_threshold:
            continue  # Skip if confidence is below threshold
        
        box = boxes[i]
        label = f"Detector {list(detectors.keys())[detector_idxs[i]]}: {confidences[i]:.2f}"

        # Draw bounding box
        cv2.rectangle(img, (box.left(), box.top()), (box.right(), box.bottom()), (0, 0, 255), 2)

        # Display label and confidence below the box
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = box.left()
        text_y = box.bottom() + text_size[1] + 5

        cv2.putText(img, label, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

    # Display the image with annotations
    cv2.imshow("Annotated Image", img)

    # Wait for a key press, press 'n' to move to the next image
    key = cv2.waitKey(0) & 0xFF
    if key == ord('n'):
        continue
    elif key == ord('q'):
        break  # Quit if 'q' is pressed

cv2.destroyAllWindows()