import cv2
import os
from deepface import DeepFace
from deepface.detectors import FaceDetector
import matplotlib.pyplot as plt
import numpy as np

# Load the reference image
reference_image_path = "me.jpg"
reference_img = cv2.imread(reference_image_path)
'''
models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "ArcFace",
    "SFace"]
backends = [
    "mtcnn",
    "opencv",
    "ssd",
    "retinaface"
]
'''
# List of backends
backends = [
    "mtcnn",
    "opencv",
    "retinaface"
]

# List of models to use
models = [
    "VGG-Face",
    "DeepFace",
    "ArcFace",
    "Facenet512"
]

# Process all images in the folder
folder_path = "D:\\stable-diffusion-webui\\outputs\\txt2img-grids\\2023-05-13-1"
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)

        for detector_backend in backends:
            # Build the face detection model
            face_detector = FaceDetector.build_model(detector_backend)

            # Detect faces in both images
            faces = FaceDetector.detect_faces(face_detector, detector_backend, img)
            reference_face = FaceDetector.detect_faces(face_detector, detector_backend, reference_img)[0][1]

            # Compare each face with the reference face and draw bounding boxes for each model
            for model in models:
                print(model)
                labeled_img = img.copy()
                scores = []
                for face in faces:
                    x, y, w, h = face[1]  # face[1] contains the region (x, y, w, h)
                    img1 = img[y:y+h, x:x+w]
                    distance_score = DeepFace.verify(img1, reference_img, model_name=model, enforce_detection=False)["distance"]
                    scores.append((distance_score, (x, y, w, h)))

                # Sort scores and select the smallest 10
                sorted_scores = sorted(scores)[:10]

                # Draw bounding boxes for the smallest 3 scores with red color
                for idx, (score, (x, y, w, h)) in enumerate(sorted_scores[:3], 1):
                    color = (0, 0, 255)  # red
                    cv2.rectangle(labeled_img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(labeled_img, f'Dis: {score:.2f}', (x-2, y-12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(labeled_img, f'Dis: {score:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(labeled_img, f'{idx}', (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Draw bounding boxes for the 4th to 10th smallest scores with blue color
                for idx, (score, (x, y, w, h)) in enumerate(sorted_scores[3:], 4):
                    color = (255, 0, 0)  # blue
                    cv2.rectangle(labeled_img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(labeled_img, f'Dis: {score:.2f}', (x-2, y-12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(labeled_img, f'Dis: {score:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(labeled_img, f'{idx}', (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Save the labeled image in the same folder with the model name and backend information
                labeled_image_path = os.path.join(folder_path, f'label-{model}-{detector_backend}-{filename}')
                cv2.imwrite(labeled_image_path, labeled_img)
