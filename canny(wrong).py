import cv2
import numpy as np
import os


def detect_edges_canny(image_path, threshold1=100, threshold2=200):
    """
    Detects edges in an image using the Canny edge detection algorithm.

    Args:
        image_path (str): The path to the input image file.
        threshold1 (int): The first threshold for the hysteresis procedure (lower bound).
        threshold2 (int): The second threshold for the hysteresis procedure (upper bound).
    """
    # 1. Check if the image file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        print("Please ensure you have an image file named 'input_image.jpg' in the same directory.")
        return

    # 2. Load the image
    img = cv2.imread(image_path)

    if img is None:
        print(
            f"Error: Could not load image from '{image_path}'. Check file integrity.")
        return

    print(f"Successfully loaded image: {image_path}")
    print(
        f"Applying Canny Edge Detection with T1={threshold1}, T2={threshold2}...")

    # 3. Convert the image to grayscale (Canny often works best on grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Apply the Canny Edge Detector
    # The Canny algorithm automatically applies a Gaussian blur before detection.
    # The two thresholds (T1 and T2) are crucial:
    # - Edges with intensity gradient > T2 are guaranteed to be edges.
    # - Edges with intensity gradient < T1 are guaranteed to be non-edges.
    # - Edges between T1 and T2 are classified as edges only if they are connected
    #   to guaranteed edges (a process called hysteresis).
    edges = cv2.Canny(gray, threshold1, threshold2)

    # 5. Display the results
    cv2.imshow('1 - Original Image', img)
    cv2.imshow('2 - Canny Edges', edges)

    print("Displaying images. Press any key to close the windows.")

    # Wait indefinitely until a key is pressed
    cv2.waitKey(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # --- Instructions ---
    # 1. Make sure you have OpenCV installed: pip install opencv-python
    # 2. Place an image in the same directory and rename it to 'input_image.jpg'.

    # Define the image path and Canny thresholds
    image_file = 'fotos/cubo6cm.jpg'

    # You can experiment with these values. Higher T1 and T2 yield fewer edges.
    # Try: (50, 150), (10, 50), or (200, 400)
    T1 = 50
    T2 = 300

    detect_edges_canny(image_file, T1, T2)
