#python script.py cpucore=4 topn=100 facefake="D:/sd.webui/webui/outputs/txt2img-images/"

import os
import sys
import cv2
import shutil
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
    img1path, facefake, model, img_count, target_img_count, total_images = args
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
                    distance_score = calculate_distance(img1, img2, model)
                    
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

def main(cpucore, topn, facefake):
    facereal = "realimage"
    model = 'Facenet512'
    faceselect = "D:/sd.webui/webui/outputs/best_matches"

    if not os.path.exists(faceselect):
        os.makedirs(faceselect)

    total_images = sum([len(files) for r, d, files in os.walk(facefake) if any(fname.endswith(('.jpg', '.png')) for fname in files)])
    target_img_count = sum([len(files) for r, d, files in os.walk(facereal) if any(fname.endswith(('.jpg', '.png')) for fname in files)])
    target_images = get_target_images(facereal)

    with Pool(processes=cpucore) as pool:
        results = pool.map(process_image, [(img1path, facefake, model, img_count, target_img_count, total_images) for img1path, img_count in target_images])
        
        avg_distance_scores = defaultdict(float)
        for result in results:
            for img_path, distance_score in result:
                avg_distance_scores[img_path] += distance_score / target_img_count

    sorted_avg_distance_scores = sorted(avg_distance_scores.items(), key=lambda x: x[1])

    for i, (img_path, avg_distance) in enumerate(sorted_avg_distance_scores[:topn]):
        new_filename = f"z000{i+1}_avg_dist_{avg_distance:.4f}.png"
        shutil.copy2(img_path, os.path.join(faceselect, new_filename))

if __name__ == "__main__":
    cpucore = 4
    topn = 100
    facefake = "D:/sd.webui/webui/outputs/txt2img-images2/"

    for arg in sys.argv[1:]:
        key, value = arg.split('=')
        if key.lower() == 'cpucore':
            cpucore = int(value)
        elif key.lower() == 'topn':
            topn = int(value)
        elif key.lower() == 'facefake':
            facefake = value

    main(cpucore, topn, facefake)
