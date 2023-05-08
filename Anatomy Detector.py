import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def wound_segmentation(image_file):
    # Load the image
    image = cv2.imread(image_file)

    # Resize the image to reduce computation time
    image = cv2.resize(image, (400, 400))

    # Convert the image to the L*a*b* color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Flatten the L*a*b* image
    lab_flat = lab.reshape((-1, 3))

    # Apply k-means clustering with 2 clusters (background and wound)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(lab_flat)

    # Assign each pixel to its corresponding cluster
    labels = kmeans.labels_.reshape(lab.shape[:2])

    # Determine the cluster with the highest average L* value (brightness)
    avg_l_values = [np.mean(lab[labels == i, 0]) for i in range(2)]
    background_label = np.argmax(avg_l_values)

    # Create a binary mask where the wound pixels are set to 255
    mask = np.where(labels == background_label, 0, 255).astype(np.uint8)

    # Perform morphological closing to fill small holes in the wound region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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


# Define image folder path
image_folder_path = "C:\\Users\\dperk\\OneDrive\\Desktop\\Medetec Medical Images"

# Create a list of image file paths
image_files = [os.path.join(image_folder_path, f) for f in os.listdir(
    image_folder_path) if f.endswith('.jpg') or f.endswith('.jpeg')]

# Loop over each image
for image_file in image_files:
    wound_segmentation(image_file)
