import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define image folder path
image_folder_path = "C:\\Users\\dperk\\OneDrive\\Desktop\\Medetec Medical Images1"

# Create a list of image file paths
image_files = [os.path.join(image_folder_path, f) for f in os.listdir(
    image_folder_path) if f.endswith('.jpg') or f.endswith('.jpeg')]

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
    pts = []
    drawing_done = False

    def draw(event, x, y, flags, param):
        global pts, drawing_done
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            pts.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            drawing_done = True

    cv2.setMouseCallback("Image", draw)

    # Wait for the user to finish drawing
    while not drawing_done:
        # Draw the current polygon on the highlight image
        if len(pts) > 1:
            cv2.polylines(highlight, [np.array(pts)],
                          True, (0, 255, 255), thickness=5)

        # Merge the original and highlighted images
        output = cv2.addWeighted(image, 0.7, highlight, 0.3, 0)

        cv2.imshow("Image", output)
        cv2.waitKey(1)

    # Create a mask image for the wound region
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Draw the final polygon on the mask
    if len(pts) > 2:
        cv2.fillPoly(mask, [np.array(pts)], 255)

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
