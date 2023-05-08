import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define image folder path
image_folder_path = "C:\\Users\\dperk\\OneDrive\\Desktop\\Medetec Medical Images"

# Create a list of image file paths
image_files = [os.path.join(image_folder_path, f) for f in os.listdir(
    image_folder_path) if f.endswith('.jpg') or f.endswith('.jpeg')]

# Loop over each image
for image_file in image_files:
    # Load the image
    image = cv2.imread(image_file)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Apply adaptive thresholding to highlight regions with significant color differences
    mask = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the mask
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask image for the wound region
    mask = np.zeros_like(mask)

    # Find the contour with the largest area (the wound region)
    max_contour = max(contours, key=cv2.contourArea)

    # Draw the contour on the mask image
    cv2.drawContours(mask, [max_contour], 0, 255, -1)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Display the original and masked images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[1].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Masked")
    plt.show()
