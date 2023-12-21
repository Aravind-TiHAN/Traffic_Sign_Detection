import cv2
import numpy as np
import os

def perform_feature_matching(reference_images_folder, classifiers_folder):
    cap = cv2.VideoCapture(1)  # Access the webcam (change the index if you have multiple webcams)

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()
    reference_images = []
    reference_keypoints_list = []
    reference_descriptors_list = []
    file_count = 0
    # Load reference images and compute keypoints and descriptors
    for filename in os.listdir(reference_images_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            reference_image_path = os.path.join(reference_images_folder, filename)
            reference_image = cv2.imread(reference_image_path)
            reference_image = cv2.resize(reference_image, (500, 500))
            reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
            reference_keypoints, reference_descriptors = sift.detectAndCompute(reference_gray, None)

            reference_images.append(reference_image)
            reference_keypoints_list.append(reference_keypoints)
            reference_descriptors_list.append(reference_descriptors)
            file_count+=1

    classifiers = []
    classifiers_count = 0
    for filename in os.listdir(classifiers_folder):
        if filename.endswith(".xml"):
            classifier_path = os.path.join(classifiers_folder, filename)
            classifier = cv2.CascadeClassifier(classifier_path)
            classifiers.append(classifier)
            classifiers_count+=1

    while True:
        ret, frame = cap.read()  # Read frame from the webcam
        if not ret:
            break

        frame = cv2.resize(frame, (500, 500))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Traffic sign detection using pre-trained classifiers
        signs = []
        for classifier in classifiers:
            detected_signs = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            signs.extend(detected_signs)

        if len(signs) > 0:
            for (x, y, w, h) in signs:
                sign_roi = gray[y:y+h, x:x+w]

                keypoints, descriptors = sift.detectAndCompute(sign_roi, None)
                image_keypoints = cv2.drawKeypoints(frame, keypoints, None)
                image_keypoints = cv2.resize(image_keypoints, (300, 300))
                cv2.imshow("org", image_keypoints)

                for i, reference_image in enumerate(reference_images):
                    reference_keypoints = reference_keypoints_list[i]
                    reference_descriptors = reference_descriptors_list[i]

                    matches = bf.knnMatch(descriptors, reference_descriptors, k=2)

                    good_matches = []
                    for m, n in matches:
                        if m.distance < 0.65 * n.distance:
                            good_matches.append(m)

                    similarity_score = len(good_matches)
                    reference_keypoints_count = len(reference_keypoints)
                    matching_threshold = reference_keypoints_count // 10

                    if similarity_score >= matching_threshold:
                        print("Traffic sign board is a match with reference image:", reference_image_path)
                    else:
                        print("No confident match for the traffic sign board.")

                    matchedImage = cv2.drawMatches(frame, keypoints, reference_image, reference_keypoints, good_matches,
                                                   None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
                    cv2.imshow("Matches", matchedImage)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit when 'q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()


reference_images_folder = "C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Sign_board_Dataset\\batch"
classifiers_folder = "C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Sign_board_Dataset\\Annotations\\Annotations"

perform_feature_matching(reference_images_folder, classifiers_folder)
