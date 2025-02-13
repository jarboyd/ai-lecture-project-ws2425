import os
import shutil
import pandas as pd
from PIL import Image


def convert_images(input_dir, output_dir, is_subdir_structure=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check if the input directory contains subdirectories or just images
    if is_subdir_structure:
        # Loop through all class subdirectories for training images
        for class_folder in os.listdir(input_dir):
            class_path = os.path.join(input_dir, class_folder)

            # Skip files if the current path is not a directory
            if not os.path.isdir(class_path):
                continue

            output_class_path = os.path.join(output_dir, class_folder)
            if not os.path.exists(output_class_path):
                os.makedirs(output_class_path)

            # Loop through each .ppm image in the class folder
            for image_file in os.listdir(class_path):
                if image_file.endswith(".ppm"):
                    img_path = os.path.join(class_path, image_file)
                    img = Image.open(img_path)
                    output_image_path = os.path.join(output_class_path, image_file.replace(".ppm", ".jpg"))
                    img.convert("RGB").save(output_image_path)
    else:
        # For test images that are directly in the folder without subdirectories
        for image_file in os.listdir(input_dir):
            if image_file.endswith(".ppm"):
                img_path = os.path.join(input_dir, image_file)
                img = Image.open(img_path)
                output_image_path = os.path.join(output_dir, image_file.replace(".ppm", ".jpg"))
                img.convert("RGB").save(output_image_path)

    print(f"Conversion complete for {input_dir} to {output_dir}.")


# Function to sort test images into their respective class folders
def sort_test_images(annotations_file, image_dir, sorted_dir):
    # Create the Sorted Images directory if it doesn't exist
    os.makedirs(sorted_dir, exist_ok=True)

    # Load the CSV with annotations
    annotations = pd.read_csv(annotations_file, sep=';')

    # Create folders and sort images based on class
    for _, row in annotations.iterrows():
        # Get the filename and class ID
        filename = row['Filename']
        class_id = row['ClassId']

        # Construct the full path to the image file
        img_path = os.path.join(image_dir, filename.replace('.ppm', '.jpg'))  # Assuming .jpg is used for images

        if os.path.exists(img_path):  # Check if the image exists
            # Create a directory for the class if it doesn't exist
            class_folder = os.path.join(sorted_dir, str(class_id))
            os.makedirs(class_folder, exist_ok=True)

            # Move or copy the image into the corresponding class folder
            shutil.move(img_path, os.path.join(class_folder, filename.replace('.ppm', '.jpg')))  # Moving the file

    print("Test images have been sorted into class folders.")


# Paths for the datasets
training_input_dir = "Datasets/GTSRB/Final_Training/Images/"
training_output_dir = "Datasets/GTSRB/Final_Training/Images_Converted/"
test_input_dir = "Datasets/GTSRB/Final_Test/Images/"
test_output_dir = "Datasets/GTSRB/Final_Test/Images_Converted/"
sorted_test_images_dir = "Datasets/GTSRB/Final_Test/Sorted_Images/"
annotations_file = "Datasets/GT-final_test.csv"

# Convert both training and test images
convert_images(training_input_dir, training_output_dir, is_subdir_structure=True)  # Training has subdirectories
convert_images(test_input_dir, test_output_dir, is_subdir_structure=False)  # Test does not have subdirectories

# After conversion, sort the test images into class folders
sort_test_images(annotations_file, test_output_dir, sorted_test_images_dir)