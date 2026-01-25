import cv2
import numpy as np
import os

# Define visualization colors for up to 9 dynamically detected clusters.
# These are fixed BGR values used only for drawing the rectangles on the output image.
VISUALIZATION_COLORS_BGR = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (0, 165, 255),  # Orange
    (255, 255, 255),  # White
    (255, 0, 255),  # Magenta/Pink
    (128, 0, 128),  # Purple
    (255, 255, 0),  # Cyan
]


def hsv_distance(hsv1, hsv2):
    """
    Calculates the Euclidean distance between two HSV points, 
    accounting for the cyclic nature of the Hue (H) component (0 to 180).
    """
    # 1. Calculate Hue distance (circular)
    h_diff = abs(hsv1[0] - hsv2[0])
    h_dist = min(h_diff, 180 - h_diff)

    # 2. Calculate Saturation and Value distance (linear)
    s_dist = abs(hsv1[1] - hsv2[1])
    v_dist = abs(hsv1[2] - hsv2[2])

    # 3. Combine using Euclidean distance (weighted to prioritize Hue, as it defines the color)
    # A 5x weight on Hue is often effective.
    return np.sqrt((h_dist * 5)**2 + s_dist**2 + v_dist**2)


def get_square_hsv(square_img):
    """
    Analyzes the square image and returns its average HSV color.
    """
    # Convert to HSV color space
    hsv_square = cv2.cvtColor(square_img, cv2.COLOR_BGR2HSV)

    # Calculate the average HSV value of the square
    h, s, v = np.mean(hsv_square, axis=(0, 1)).astype(int)
    return (h, s, v)


def cluster_hsv_colors(hsv_values, threshold_distance=45):
    """
    Groups the 9 detected HSV values into clusters based on proximity using a simple 
    iterative clustering method (similar to k-means but dynamically creating clusters).
    """
    clusters = []  # Stores (center_hsv, [assigned_hsv_values])
    assignments = []  # Stores the name for each input HSV value

    for hsv_value in hsv_values:
        best_cluster_index = -1
        min_distance = float('inf')

        # Find the closest existing cluster center
        for i, (center_hsv, _) in enumerate(clusters):
            dist = hsv_distance(hsv_value, center_hsv)
            if dist < min_distance:
                min_distance = dist
                best_cluster_index = i

        # Check if the closest cluster is within the acceptable threshold
        if min_distance <= threshold_distance:
            # Assign to existing cluster
            cluster_name = f"Color {chr(65 + best_cluster_index)}"
            assignments.append(cluster_name)

            # Update cluster: add the new point and re-calculate the center
            clusters[best_cluster_index][1].append(hsv_value)
            new_center = tuple(
                np.mean(clusters[best_cluster_index][1], axis=0).astype(int))
            clusters[best_cluster_index] = (
                new_center, clusters[best_cluster_index][1])

        else:
            # Start a new cluster
            new_cluster_name = f"Color {chr(65 + len(clusters))}"
            assignments.append(new_cluster_name)
            clusters.append((hsv_value, [hsv_value]))

    return assignments, [c[0] for c in clusters]


def find_and_warp_cube_face(img, target_size=450):
    """
    Detects the main cube face in the image, corrects its perspective, 
    and returns a clean, square image of the face.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce high-frequency noise and improve Canny results
   # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny to find strong edges
    # T1=50, T2=150 are good starting values for general edge detection
    edges = cv2.Canny(img, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Target contour for the cube face
    cube_contour = None
    max_area = 0

    for contour in contours:
        # Approximate the contour shape to reduce the number of points
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        # We are looking for quadrilaterals (4 points) that are large
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                cube_contour = approx

    if cube_contour is None:
        raise ValueError(
            "Could not find a suitable quadrilateral (cube face) in the image.")

    # Reformat the corners for perspective transform
    # The points must be sorted in a consistent order (e.g., top-left, top-right, bottom-right, bottom-left)

    # Reshape to (4, 2)
    pts = cube_contour.reshape(4, 2)

    # Determine the order of points:
    # 1. Sum of coordinates gives the top-left (min sum) and bottom-right (max sum)
    # 2. Difference of coordinates gives the top-right (min diff) and bottom-left (max diff)
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left
    rect[2] = pts[np.argmax(s)]  # Bottom-Right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left

    # Define the destination points for the warp (a perfect square)
    dst = np.array([
        [0, 0],                            # Top-Left
        [target_size - 1, 0],              # Top-Right
        [target_size - 1, target_size - 1],  # Bottom-Right
        [0, target_size - 1]],             # Bottom-Left
        dtype="float32"
    )

    # Get the perspective transformation matrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # Apply the warp transformation
    warped_img = cv2.warpPerspective(img, M, (target_size, target_size))

    # Optional: Draw the detected contour on the original image for debugging
    cv2.drawContours(img, [cube_contour], -1, (0, 255, 0), 5)

    return warped_img, img


def analyze_rubiks_cube_face(image_path):
    """
    Loads an image, finds the cube face, corrects perspective, divides it into 
    9 squares, and determines the color of each square dynamically via clustering.

    Args:
        image_path (str): Path to the image file (e.g., 'cube_face.jpg').
    """
    # 1. Check file existence
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        print("Please ensure you have an image file named 'cube_face.jpg' in the same directory.")
        return

    # 2. Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(
            f"Error: Could not load image from '{image_path}'. Check file integrity.")
        return

    print(f"\n1. Finding and correcting perspective for the cube face...")
    try:
        # 3. Find cube face, warp it to a perfect square, and get the visualization image
        warped_img, visualization_img = find_and_warp_cube_face(img)
        cv2.imshow('1 - Original Image (Outline)', visualization_img)
        cv2.imshow('2 - Warped Cube Face (Color Analysis)', warped_img)
        print("\nDisplaying images. Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    except ValueError as e:
        print(f"Error during cube detection: {e}")
        return

    H, W, _ = warped_img.shape

    # 4. Determine square dimensions on the cleaned, warped image
    margin_percent = 0.05
    square_h = H // 3
    square_w = W // 3

    all_hsv_values = []
    square_data = []  # Stores (crop_coords, hsv_value)

    print(f"2. Analyzing the {H}x{W} warped grid divided into 3x3 squares...")

    # 5. Extract the average HSV value for each of the 9 squares
    for r in range(3):
        for c in range(3):
            start_y, end_y = r * square_h, (r + 1) * square_h
            start_x, end_x = c * square_w, (c + 1) * square_w

            # Apply margin to crop the center area of the square (avoiding borders/gaps)
            margin_y, margin_x = int(
                square_h * margin_percent), int(square_w * margin_percent)
            crop_start_y, crop_end_y = start_y + margin_y, end_y - margin_y
            crop_start_x, crop_end_x = start_x + margin_x, end_x - margin_x

            square_segment = warped_img[crop_start_y:crop_end_y,
                                        crop_start_x:crop_end_x]
            avg_hsv = get_square_hsv(square_segment)
            all_hsv_values.append(avg_hsv)

            square_data.append({
                'r': r, 'c': c,
                'hsv': avg_hsv,
                'coords': (crop_start_x, crop_start_y, crop_end_x, crop_end_y)
            })

    # 6. Cluster the 9 HSV values dynamically
    color_assignments, cluster_centers = cluster_hsv_colors(
        all_hsv_values, threshold_distance=45)

    num_clusters = len(cluster_centers)
    print(f"3. Identified {num_clusters} distinct color clusters.")

    # 7. Prepare the final grid and visualization on the warped image
    result_grid = [[''] * 3 for _ in range(3)]

    for i, data in enumerate(square_data):
        r, c = data['r'], data['c']
        color_name = color_assignments[i]

        # Get the cluster index (A=0, B=1, ...) to find the visualization color
        cluster_index = ord(color_name.split(' ')[1]) - 65

        # Determine the BGR color for the rectangle
        color_bgr = VISUALIZATION_COLORS_BGR[cluster_index % len(
            VISUALIZATION_COLORS_BGR)]

        result_grid[r][c] = color_name

        # Draw the colored rectangle on the WARPED image
        start_x, start_y, end_x, end_y = data['coords']
        cv2.rectangle(warped_img, (start_x, start_y),
                      (end_x, end_y), color_bgr, 5)

    # 8. Print the results
    print("\n--- Dynamic Rubik's Cube Face Color Assignment (Row by Row) ---")
    for row in result_grid:
        print(f"| {row[0]:<10} | {row[1]:<10} | {row[2]:<10} |")
    print("-------------------------------------------------------------")

    print("\n--- Identified Color Cluster Centers (H, S, V) ---")
    for i, center in enumerate(cluster_centers):
        print(f"Color {chr(65 + i)}: HSV {center}")
    print("-------------------------------------------------")

    # 9. Display the original image (with detected outline) and the warped result
    cv2.imshow('1 - Original Image (Outline)', visualization_img)
    cv2.imshow('2 - Warped Cube Face (Color Analysis)', warped_img)
    print("\nDisplaying images. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # --- Instructions ---
    # 1. Make sure you have OpenCV and NumPy installed: pip install opencv-python numpy
    # 2. Place an image of a Rubik's Cube face (it can now have a background and be at an angle)
    #    in the same directory and rename it to 'cube_face.jpg'.

    image_file = 'fotos/cubo6cm.jpg'
    analyze_rubiks_cube_face(image_file)
