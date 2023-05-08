import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define image folder path
image_folder_path = "C:\\Users\\dperk\\OneDrive\\Desktop\\Medetec Medical Images1"

# Create a list of image file paths
image_files = [os.path.join(image_folder_path, f) for f in os.listdir(
    image_folder_path) if f.endswith('.jpg') or f.endswith('.png')]

# Loop over each image
for image_file in image_files:
    # Load the image
    image = cv2.imread(image_file)

    # Create a copy of the image to draw on
    drawing = image.copy()

    # Create a window to display the image
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", (800, 800))
    cv2.imshow("Image", drawing)

    # Create a copy of the original image for highlighting the drawing
    highlight = image.copy()

    # Set up the mouse callback function for drawing
    drawing_done = False
    prev_pt = None

    def draw(event, x, y, flags, param):
        global prev_pt, drawing_done
        if event == cv2.EVENT_LBUTTONDOWN:
            prev_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            if prev_pt is not None:
                cv2.line(drawing, prev_pt, (x, y), (0, 0, 255), thickness=5)
                prev_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing_done = True
            if prev_pt is not None:
                cv2.line(drawing, prev_pt, (x, y), (0, 0, 255), thickness=5)
                # Add this line to draw on the mask
                cv2.line(mask, prev_pt, (x, y), 255, thickness=5)

    cv2.setMouseCallback("Image", draw)

    # Wait for the user to finish drawing
    while not drawing_done:
        cv2.imshow("Image", drawing)
        cv2.waitKey(1)

    # Create a mask image for the wound region
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Display the original and masked images
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[1].imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Masked")
    axs[2].imshow(mask, cmap="gray")
    axs[2].set_title("Mask")
    plt.show()
