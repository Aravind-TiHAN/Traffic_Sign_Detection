import cv2
import numpy as np


def differentiate_lanes(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for red and green colors in HSV
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    lower_blue = np.array([0, 0, 150])
    upper_blue = np.array([100, 100, 255])
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])

    # Threshold the image to extract the red and green regions
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

    red_color = cv2.bitwise_and(image, image, mask=red_mask)
    blue_color = cv2.bitwise_and(image, image, mask=blue_mask)
    green_color = cv2.bitwise_and(image, image, mask=green_mask)


    # Find contours in the red and green masksq
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contours, _ = cv2.findContours(blue_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_mean_color = cv2.mean(red_color)[:3]  # Mean color values of the red regions (B, G, R)
    blue_mean_color = cv2.mean(blue_color)[:3]  # Mean color values of the blue regions (B, G, R)
    green_mean_color = cv2.mean(green_color)[:3]  # Mean color values of the green regions (B, G, R)

    # Draw the detected red and green lanes on the original image
    for contour in red_contours:
        cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
    for contour in blue_contours:
        cv2.drawContours(image,[contour], -1,(255,0,0),2)
    for contour in green_contours:
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    # Calculate the width of the green lane
    # if len(green_contours) > 0:
    #     # Get the largest green contour
    #     largest_contour = max(green_contours, key=cv2.contourArea)
    #
    #     # Calculate the bounding rectangle around the contour
    #     x, y, w, h = cv2.boundingRect(largest_contour)
    #
    #     # Draw the bounding rectangle around the green lane
    #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    #     # Calculate the width of the green region
    #     green_width = w
    #
    #     # Print the width of the green region
    #     print("Width of the green region:", green_width)

    # Display the resulting image
    print("Red color:", red_mean_color)
    print("Blue color:", blue_mean_color)
    print("Green color:", green_mean_color)
    cv2.imshow('red_mask',red_mask)
    cv2.imshow("Differentiated Lanes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Path to the input image
image_path = 'C:\\Users\\mekal\\OneDrive\\Desktop\\Traffic sign det\\venv\\Cropped_images\\1690004553.png'

# Differentiate lanes based on color and find the width of the green region
differentiate_lanes(image_path)
