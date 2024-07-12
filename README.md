# Face_Exp_Naveen
Real-Time Facial Expression Recognition(Face_Exp_Naveen)
# Real-Time Facial Expression Recognition

## Introduction
This project demonstrates a real-time facial expression recognition system using deep learning. The system captures video from a webcam, detects faces, and classifies their expressions into categories such as Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Dataset
The dataset used for training the model is the FER-2013 (Facial Expression Recognition 2013) dataset. This dataset contains 48x48 pixel grayscale images of faces, each labeled with one of seven emotion categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Steps

1. **Data Preparation**: 
   - Load and preprocess the FER-2013 dataset.
   - Normalize the pixel values and reshape the images to 48x48 pixels.

2. **Model Training**: 
   - Define a Convolutional Neural Network (CNN) architecture.
   - Compile the model with categorical cross-entropy loss and the Adam optimizer.
   - Train the model on the training data and validate it on the validation data.

3. **Model Saving**: 
   - Save the trained model to an HDF5 file for later use.

4. **Real-Time Recognition**: 
   - Load the saved model.
   - Capture video from a webcam and preprocess each frame.
   - Detect faces in the frame using OpenCV's Haar Cascade.
   - Predict the emotion of each detected face using the loaded model.
   - Display the detected faces and their predicted emotions on the video feed.

## New Model
The new model created is a Convolutional Neural Network (CNN) designed to classify facial expressions in real-time. The model consists of multiple convolutional layers, max-pooling layers, dropout layers, and dense layers. This architecture helps the model to learn and recognize complex patterns in facial expressions, making it effective for real-time applications. The model achieves good accuracy on the FER-2013 dataset and can be further improved with more data and fine-tuning.

## Installation

To run this project, you need Python installed along with the following libraries:
- `tensorflow`
- `opencv-python`
- `numpy`

You can install these libraries using pip:

```sh
pip install tensorflow opencv-python numpy
