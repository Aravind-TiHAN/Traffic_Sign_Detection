import cv2
import numpy as np
from scipy.spatial import distance
import easyocr
import os
import time


def load_image(image_path):
    image = cv2.imread(image_path)
    return image


def detect_color(image):
    # Your color detection code here
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the color ranges for blue, green, and red
    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([120, 255, 255])

    lower_green = np.array([40, 100, 100])
    upper_green = np.array([70, 255, 255])

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    # Create masks for blue, green, and red color ranges
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combine the masks
    combined_mask = cv2.bitwise_or(mask_blue, mask_green)
    combined_mask = cv2.bitwise_or(combined_mask, mask_red1)
    combined_mask = cv2.bitwise_or(combined_mask, mask_red2)

    # Apply the combined mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=combined_mask)
    average_color = np.mean(masked_image, axis=(1, 0))
    det_color = np.argmax(average_color)

    color = ""
    if det_color == 0:
        color = "BLUE"
    if det_color == 1:
        color = "GREEN"
    if det_color == 2:
        color = "RED"
    # print('Detected color:',color)

    # ...
    # color = (0, 0, 255)  # Sample color for demonstration purposes
    return color


def detect_shape(image):
    # Your shape detection code here
    # ...
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((4, 4), np.uint8)
    dilation = cv2.dilate(gray, kernel, iterations=1)
    blur = cv2.GaussianBlur(dilation, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=0,
                               maxRadius=0)
    shape = " "
    if circles is None:
        # Find contours in the image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Iterate through the contours and approximate the shape
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.0005 * perimeter, True)

            # Determine the number of vertices of the shape
            num_vertices = len(approx)

            if num_vertices == 3:
                shape = "Triangle"
            elif num_vertices == 4:
                shape = "Rectangle"
            elif num_vertices == 5:
                shape = "Pentagon"
            elif num_vertices == 6:
                shape = "Hexagon"
            elif num_vertices == 7:
                shape = "Septagon"
            elif num_vertices == 8:
                shape = "Octagon"
            else:
                shape = "Other"

            # Draw the shape contour
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
    else:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            shape = "Circle"

    # shape = "Circle"  # Sample shape for demonstration purposes
    return shape


def extract_text_from_image(image):
    # Your text extraction code here

    img = cv2.resize(image, (500, 500))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 1)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img, detail=1, paragraph=True)
    out_text = reader.readtext(img, detail=0)
    for (coord, text) in result:
        (topleft, topright, bottomright, bottomleft) = coord
        tx, ty = (int(topleft[0]), int(topleft[1]))
        bx, by = (int(bottomright[0]), int(bottomright[1]))
        cv2.rectangle(img, (tx, ty), (bx, by), (0, 0, 255), 2)
        cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # ...
    # text = "STOP"  # Sample text for demonstration purposes
    return out_text


# def find_closest_color(color, color_map):
#     # Your find_closest_color function here
#     # ...
#     closest_color = "Red"  # Sample color name for demonstration purposes
#     return closest_color

def perform_feature_matching(board_image_path, reference_images_path):
    # Your feature matching code here
    # ...
    board_image = cv2.imread(board_image_path)
    board_image = cv2.resize(board_image, (500, 500))
    board_gray = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)

    # sift = cv2.SIFT_create()
    # bf = cv2.BFMatcher()
    kaze = cv2.KAZE_create()
    flann = cv2.FlannBasedMatcher()

    board_keypoints, board_descriptors = kaze.detectAndCompute(board_gray, None)
    image_keypoints2 = cv2.drawKeypoints(board_image, board_keypoints, None)
    image_keypoints2 = cv2.resize(image_keypoints2, (300, 300))

    max_similarity_score = 0
    reference_image_path_with_highest_matching = ""

    # reference_image_path_with_highest_matching = ""  # Sample result for demonstration purposes
    # return reference_image_path_with_highest_matching

    for reference_image_path in reference_images_path:
        reference_image = cv2.imread(reference_image_path)
        reference_image = cv2.resize(reference_image, (500, 500))
        reference_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        reference_keypoints, reference_descriptors = kaze.detectAndCompute(reference_image, None)
        image_keypoints1 = cv2.drawKeypoints(reference_image, reference_keypoints, None)
        image_keypoints1 = cv2.resize(image_keypoints1, (300, 300))

    matches = flann.knnMatch(board_descriptors, reference_descriptors, k=2)
    matched = ''
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    similarity_score = len(good_matches)
    reference_keypoints_count = len(reference_keypoints)

    matching_threshold = reference_keypoints_count // 10

    if similarity_score > max_similarity_score:
        max_similarity_score = similarity_score
        reference_image_path_with_highest_matching = reference_image_path

    matchedImage = cv2.drawMatches(board_image, board_keypoints, reference_image, reference_keypoints,
                                   good_matches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

    if reference_image_path_with_highest_matching != "":
        return reference_image_path_with_highest_matching
    else:
        print("No confident match for the traffic sign board.")
    return matched


def main():
    # Step 1: Load the input images from the folder
    images_folder_path = 'C:\\Users\\mekal\\RealTime_Detection-And-Classification-of-TrafficSigns\\Codes'
    image_files = [os.path.join(images_folder_path, f) for f in os.listdir(images_folder_path) if
                   f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_path in image_files:
        print("Processing image:", image_path)
        # Load the input image
        image = load_image(image_path)

        # Step 2: Color Detection
        extracted_text = extract_text_from_image(image)

        if extracted_text == '[]':
            color = detect_color(image)
            print("Color:", color)

            shape = detect_shape(image)
            print("Shape:", shape)

        # Step 5: Feature Matching
            board_image_path = image_path

            reference_images_path = []
            reference_folder_path = "C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Mandatory_Signs\\"
            for filename in os.listdir(reference_folder_path):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    image_path = os.path.join(reference_folder_path, filename)
                    reference_images_path.append(image_path)
        # reference_images_path = ['C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Barrier Ahead.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Cattle.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Cross Road.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Cycle Crossing.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Dangerous Dip.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Falling Rocks.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Ferry.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Gap in Median.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Guarded Level Crossing .png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Hump or Rough Road.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Left Hair Pin Bend.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Left Hand Curve.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Left Reverse Bend.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Loose Gravel.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Men at Work.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Narrow Bridge.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Narrow Road Ahead.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Pedestrain Crossing.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Quayside or River Bank.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Right Hand Curve.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Right Hir Pin Bend.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Right Reverse Bend.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Road Widens Ahead.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Round About.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\School Ahead.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Side Road Left.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Side Road Right.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Slippery Road.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Staggered Intersection.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Steep Ascent.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Steep Desent .png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\T- Intersection.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Traffic Signal.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Unguarded Level Crossing .png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Y- Intersection to left.png', 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cautionary Road Signs\\Y- Intersection.png']
                    reference_image_path_with_highest_matching = perform_feature_matching(board_image_path, reference_images_path)
                    print("Matching Reference Image:", reference_image_path_with_highest_matching)
        else:
            print("Extracted Text:", extracted_text)

        # Display the outputs and images
        cv2.imshow('Original Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
