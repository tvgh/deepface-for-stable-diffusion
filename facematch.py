"""
This Python script is designed to find and select the best-matching images of fake faces based on their distance from real faces.

The script starts by importing necessary libraries, such as os, sys, cv2, shutil, time, collections, and multiprocessing. It also imports the DeepFace library for facial recognition tasks.

The read_image function reads an image from a given file path and returns the image object. If an error occurs, it prints the error message and returns None.

The calculate_distance function takes two images and a model as input, and returns the distance between the two images using the DeepFace.verify function. If an error occurs, it prints the error message and returns None.

The process_image function takes several arguments, such as image paths, facefake folder, model, image counts, etc. It reads the target image and calculates the distance between the target image and images in the facefake folder. It adds the distance scores to a list and returns the list once the processing is done.

The get_target_images function takes a facereal folder as input and returns a list of target images found in the folder.

The main function initializes some variables, such as cpucore, maxdistance, facefake, and facereal folders. It also calculates the total number of images to process and creates a target_img_count variable to store the total count of target images found.

It then uses multiprocessing with a Pool to parallelize the processing of images. The results are stored in a defaultdict with float data type to calculate the average distance scores.

Next, it sorts the average distance scores and selects the images with a distance score less than or equal to the maxdistance variable.

Finally, it copies the selected images to a new folder, faceselect, and prints the script's running time.

The script also accepts command-line arguments for cpucore, maxdistance, and facefake variables, allowing users to customize the script's behavior.

call with parameter
python facematch.py cpucore=8 maxdistance=0.5 facefake="D:/stable-diffusion-webui/outputs/txt2img-images/2023-05-14"

'''

import os
import sys
import cv2
import shutil
import time
from collections import defaultdict
from multiprocessing import Pool
from deepface import DeepFace

def read_image(img_path):
    try:
        return cv2.imread(img_path)
    except Exception as e:
        print(f"Error occurred while reading {img_path}: {str(e)}")
        return None

def calculate_distance(img1, img2, model, backend):
    try:
        return DeepFace.verify(img1, img2, model_name=model, detector_backend=backend)["distance"]
    except Exception as e:
        print(f"Error occurred while processing: {str(e)}")
        return None

def process_image(args):
    img1path, facefake, model, backend, img_count, target_img_count, total_images = args
    img1 = read_image(img1path)
    if img1 is None:
        return []

    count = 0
    distance_scores = []
    for root, _, files in os.walk(facefake):
        for filename in files:
            if filename.endswith(('.jpg', '.png')):
                img_path = os.path.join(root, filename)
                img2 = read_image(img_path)
                
                if img2 is not None:
                    distance_score = calculate_distance(img1, img2, model, backend)
                    
                    if distance_score is not None and distance_score != 0:
                        distance_scores.append((img_path, distance_score))
                        count += 1
                        print(f"Processing target image {img_count}/{target_img_count}, image {count}/{total_images}, distance: {distance_score}")

                if count == 50000:
                    break

    return distance_scores

def get_target_images(facereal):
    target_images = []
    img_count = 0
    for root, _, files in os.walk(facereal):
        for filename in files:
            if filename.endswith(('.jpg', '.png')):
                img_count += 1
                img1path = os.path.join(root, filename)
                target_images.append((img1path, img_count))
    return target_images

def main(cpucore, maxdistance, facefake, backend, model):
    facereal = "facereal"
    faceselect = facefake.replace("-images", f"-images-best-{backend}-{model}")

    if not os.path.exists(faceselect):
        os.makedirs(faceselect)

    total_images = sum([len(files) for r, d, files in os.walk(facefake) if any(fname.endswith(('.jpg', '.png')) for fname in files)])
    target_img_count = sum([len(files) for r, d, files in os.walk(facereal) if any(fname.endswith(('.jpg', '.png')) for fname in files)])
    target_images = get_target_images(facereal)

    with Pool(processes=cpucore) as pool:
        results = pool.map(process_image, [(img1path, facefake, model, backend, img_count, target_img_count, total_images) for img1path, img_count in target_images])
        
        avg_distance_scores = defaultdict(float)
        for result in results:
            for img_path, distance_score in result:
                avg_distance_scores[img_path] += distance_score / target_img_count

    sorted_avg_distance_scores = sorted(avg_distance_scores.items(), key=lambda x: x[1])
    selected_images = [(img, distance) for img, distance in sorted_avg_distance_scores if distance <= maxdistance]
    
    for i, (img_path, avg_distance) in enumerate(selected_images):
        file_ext = os.path.splitext(img_path)[1]
        new_filename = f"_face_{int(avg_distance * 10000):04d}{file_ext}"        
        shutil.copy2(img_path, os.path.join(faceselect, new_filename))

if __name__ == "__main__":
    start_time = time.time()
    cpucore = 4
    maxdistance = 0.5
    facefake = "D:/sd.webui/webui/outputs/txt2img-images/"

    for arg in sys.argv[1:]:
        key, value = arg.split('=')
        if key.lower() == 'cpucore':
            cpucore = int(value)
        elif key.lower() == 'maxdistance':
            maxdistance = float(value)
        elif key.lower() == 'facefake':
            facefake = value.replace("\\", "/")  # replace \ with /

    backends = [
        "mtcnn",
        "opencv",
        "retinaface"
    ]

    models = [
        "VGG-Face",
        "DeepFace",
        "ArcFace",
        "Facenet512"
    ]

    for backend in backends:
        for model in models:
            print(f"Processing with backend: {backend} and model: {model}")
            main(cpucore, maxdistance, facefake, backend, model)

    minutes, seconds = divmod((time.time() - start_time), 60)
    print(f"Script running time: {int(minutes)}:{int(seconds)} minutes")


