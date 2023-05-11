#python script.py "D:/sd.webui/webui/outputs/txt2img-images/" 20
import os
import cv2
import shutil
import sys
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

def calculate_distance(img1, img2, model):
    try:
        return DeepFace.verify(img1, img2, model_name=model)["distance"]
    except Exception as e:
        print(f"Error occurred while processing: {str(e)}")
        return None

def process_image(args):
    img1path, imgsfolder, model, img_count, target_img_count, total_images = args
    img1 = read_image(img1path)
    if img1 is None:
        return []

    count = 0
    distance_scores = []
    for root, _, files in os.walk(imgsfolder):
        for filename in files:
            if filename.endswith(('.jpg', '.png')):
                img_path = os.path.join(root, filename)
                img2 = read_image(img_path)
                
                if img2 is not None:
                    distance_score = calculate_distance(img1, img2, model)
                    
                    if distance_score is not None and distance_score != 0:
                        distance_scores.append((img_path, distance_score))
                        count += 1
                        print(f"Processing target image {img_count}/{target_img_count}, image {count}/{total_images}, distance: {distance_score}")

                if count == 50000:
                    break

    return distance_scores

def get_target_images(target_folder):
    target_images = []
    img_count = 0
    for root, _, files in os.walk(target_folder):
        for filename in files:
            if filename.endswith(('.jpg', '.png')):
                img_count += 1
                img1path = os.path.join(root, filename)
                target_images.append((img1path, img_count))
    return target_images

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py imgsfolder n_best_matches")
        sys.exit(1)

    imgsfolder = sys.argv[1]
    n_best_matches = int(sys.argv[2])

    target_folder = "realimage"
    model = 'Facenet512'
    destination_folder = "D:/sd.webui/webui/outputs/best_matches"
    processnum = 4

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    total_images = sum([len(files) for r, d, files in os.walk(imgsfolder) if any(fname.endswith(('.jpg', '.png')) for fname in files)])
    target_img_count = sum([len(files) for r, d, files in os.walk(target_folder) if any(fname.endswith(('.jpg', '.png')) for fname in files)])
    target_images = get_target_images(target_folder)

    start_time = time.time()

    with Pool(processes=processnum) as pool:
        results = pool.map(process_image, [(img1path, imgsfolder, model, img_count, target_img_count, total_images) for img1path, img_count in target_images])
        
        avg_distance_scores = defaultdict(float)
        for result in results:
            for img_path, distance_score in result:
                avg_distance_scores[img_path] += distance_score / target_img_count

    sorted_avg_distance_scores = sorted(avg_distance_scores.items(), key=lambda x: x[1])

    for i, (img_path, _) in enumerate(sorted_avg_distance_scores[:n_best_matches]):
        new_filename = f"z{str(i+1).zfill(5)}.png"
        shutil.copy2(img_path, os.path.join(destination_folder, new_filename))

    end_time = time.time()
    print(f"Total running time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()

 
