# Traffic Sign Recognition

This project focuses on traffic sign classification using a Convolutional Neural Network (CNN) trained on the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. The final model takes an image of a traffic sign as input and outputs its classification.

## Requirements

To run this project, you need the following Python packages:

```bash
pip install -r requirements.txt
```

## Setup

1. **Download and Prepare Data:**
   Run `download.py` and `prepare.py` to download the dataset and prepare the data for training.

2. **Train the Model (Optional):**
   Use `train.ipynb` to train the model. This step is optional if you already have a pre-trained model.

3. **Classify Images:**
   Use `predict.ipynb` to classify images using the trained model.

## Training the Model

To train the model, follow these steps:

1. Open `train.ipynb` in your preferred Jupyter Notebook environment.
2. Execute the cells to build, compile, and train the model.
3. The model will be saved at specified checkpoints during training.

## Predicting Traffic Signs

To classify traffic sign images:

1. Open `predict.ipynb` in your preferred Jupyter Notebook environment.
2. Load the pre-trained model.
3. Load and preprocess the image you want to classify.
4. Run the prediction cell to get the classification result.

## Pre-trained Model

`final_model.h5` is the pre-trained model that can be used to classify traffic signs.