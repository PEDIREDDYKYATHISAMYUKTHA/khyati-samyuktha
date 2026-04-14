import os
import cv2
import mediapipe as mp

# -------- DATASET PATH --------
dataset_path = "datasets/training_set"

# -------- MEDIAPIPE FACE DETECTOR --------
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.6
)

# -------- PROCESS DATASET --------
for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    print(f"\nProcessing class: {folder}")

    for file in os.listdir(folder_path):

        img_path = os.path.join(folder_path, file)

        try:
            image = cv2.imread(img_path)

            if image is None:
                print("Deleting corrupt image:", img_path)
                os.remove(img_path)
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = face_detector.process(rgb)

            # ---------- REMOVE IMAGES WITH NO FACE ----------
            if not results.detections:
                print("No face detected, removing:", img_path)
                os.remove(img_path)
                continue

            # ---------- CROP FACE ----------
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            h, w, _ = image.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            face = image[y:y+height, x:x+width]

            if face.size == 0:
                os.remove(img_path)
                continue

            # ---------- RESIZE ----------
            face = cv2.resize(face, (224,224))

            # ---------- OVERWRITE IMAGE ----------
            cv2.imwrite(img_path, face)

        except Exception as e:
            print("Error:", img_path)
            os.remove(img_path)

print("\nDataset cleaning completed.")