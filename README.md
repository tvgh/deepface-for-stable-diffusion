# deepface-for-stable-diffusion

get the most similar faces in stable fussion(automatic1111) txt2img folder with a few reference images using deepface model, number them by similarity and copy them to a new folder.

facelabel.py

# Face Comparison Tool

This Python script is a face comparison tool that uses several face recognition models and backends to compare faces detected in a set of images against a reference image. The purpose of this tool is to identify the most similar faces to the reference face and draw bounding boxes around them, ranking them based on their similarity scores.

## Features

- Uses multiple face detection backends such as MTCNN, OpenCV, and RetinaFace
- Utilizes various face recognition models, including VGG-Face, DeepFace, ArcFace, and Facenet512
- Processes all images in a specified folder and compares detected faces with a reference image
- Draws bounding boxes around the top 10 most similar faces and labels them with their similarity scores
- Saves labeled images with model and backend information in the filename

## How It Works

1. Load the reference image and set the list of backends and models to use.
2. Iterate through all images in the specified folder.
3. For each image, detect faces using the different backends and models.
4. Compare the detected faces with the reference face and calculate their similarity scores.
5. Sort the scores and select the top 10 most similar faces.
6. Draw bounding boxes and labels for the top 3 most similar faces in red and the remaining faces in blue.
7. Save the labeled images in the original folder with the model and backend information included in the filename.

## Requirements

- OpenCV
- DeepFace
- numpy
- matplotlib

## Who is this for?

This face comparison tool is suitable for anyone working with facial recognition, particularly in applications where it is necessary to identify the most similar faces to a reference image. This can be useful for law enforcement, security systems, or even entertainment purposes.

## Usage

To use this tool, simply set the path to the reference image, specify the folder containing the images to compare, and choose the backends and models you want to use for face detection and recognition. Then, run the script to process all images in the folder and save the labeled results.


deepface is not thread safe , so use multiprocessing instead of multithreading


