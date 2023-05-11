
import os
import cv2
import shutil
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from deepface import DeepFace

def calculate_distances(img1, imgsfolder, model, img_count, target_img_count, total_images):
    img1 = cv2.imread(img1)
    distance_scores = []
    count = 0
    for root, _, files in os.walk(imgsfolder):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                try:
                    img_path = os.path.join(root, filename)
                    img2 = cv2.imread(img_path)
                    distance_score = DeepFace.verify(img1, img2, model_name=model)["distance"]
                    distance_scores.append((img_path, distance_score))
                    count += 1
                    print(f"Processing target image {img_count}/{target_img_count}, image {count}/{total_images}, distance: {distance_score}")
                except Exception as e:
                    print("Error occurred while processing " + filename + ": " + str(e))

                if count == 50000:
                    break
    return distance_scores

def process_target_image(img1path, imgsfolder, model, img_count, target_img_count, total_images, sum_distance_scores, lock):
    distance_scores = calculate_distances(img1path, imgsfolder, model, img_count, target_img_count, total_images)
    with lock:  
        for img_path, distance_score in distance_scores:
            sum_distance_scores[img_path] += distance_score

def main():
    target_folder = "realimage"
    imgsfolder = "D:/sd.webui/webui/outputs/txt2img-images2/"
    model = 'Facenet512'
    n_best_matches = 40
    destination_folder = "D:/sd.webui/webui/outputs/best_matches"
    global_parameter = 4

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    total_images = sum([len(files) for r, d, files in os.walk(imgsfolder) if any(fname.endswith('.jpg') or fname.endswith('.png') for fname in files)])

    sum_distance_scores = defaultdict(float)
    target_img_count = 0
    for root, _, files in os.walk(target_folder):
        target_img_count += sum(1 for filename in files if filename.endswith('.jpg') or filename.endswith('.png'))

    img_count = 0
    target_images = []
    for root, _, files in os.walk(target_folder):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_count += 1
                img1path = os.path.join(root, filename)
                target_images.append((img1path, img_count))

    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=global_parameter) as executor:
        futures = [executor.submit(process_target_image, img1path, imgsfolder, model, img_count, target_img_count, total_images, sum_distance_scores, lock) for img1path, img_count in target_images]
        _ = [future.result() for future in futures]

    sorted_sum_distance_scores = sorted(sum_distance_scores.items(), key=lambda x: x[1])

    for i, (img_path, _) in enumerate(sorted_sum_distance_scores[:n_best_matches]):
        new_filename = f"z000{i+1}.png"
        shutil.copy2(img_path, os.path.join(destination_folder, new_filename))

if __name__ == "__main__":
    main()

