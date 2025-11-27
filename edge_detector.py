import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_lines(image, canny_threshold1=50, canny_threshold2=150,
               hough_threshold=100, min_line_length=50, max_line_gap=10,
               angle_tolerance_deg=10):
    """
    Find horizontal and vertical lines using HoughLinesP.
    Returns two lists: horizontal_lines, vertical_lines
    Each line is (x1, y1, x2, y2).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)

    # Probabilistic Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    horizontal_lines = []
    vertical_lines = []

    if lines is not None:
        angle_tol = np.deg2rad(angle_tolerance_deg)

        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = x2 - x1
            dy = y2 - y1

            # Avoid division by zero
            angle = np.arctan2(dy, dx)  # angle in radians

            # Normalize angle to [0, pi)
            if angle < 0:
                angle += np.pi

            # Horizontal: angle ~ 0 or ~pi
            if (abs(angle) < angle_tol) or (abs(angle - np.pi) < angle_tol):
                horizontal_lines.append((x1, y1, x2, y2))
            # Vertical: angle ~ pi/2
            elif abs(angle - np.pi/2) < angle_tol:
                vertical_lines.append((x1, y1, x2, y2))

    return horizontal_lines, vertical_lines


def get_intersections(horizontal_lines, vertical_lines):
    """
    Compute all intersection points between horizontal and vertical line segments.
    Returns a list of (x, y) tuples.
    """
    intersections = []

    for x1h, y1h, x2h, y2h in horizontal_lines:
        # For an (approximately) horizontal line, y is roughly constant,
        # but we use endpoints as segment bounds.
        yh = (y1h + y2h) / 2.0
        xh_min, xh_max = sorted([x1h, x2h])

        for x1v, y1v, x2v, y2v in vertical_lines:
            # For an (approximately) vertical line, x is roughly constant
            xv = (x1v + x2v) / 2.0
            yv_min, yv_max = sorted([y1v, y2v])

            # Intersection candidate:
            #   x = xv (vertical), y = yh (horizontal)
            xi, yi = xv, yh

            # Check if (xi, yi) lies within both segments

            intersections.append((int(round(xv)), int(round(yh))))

    return intersections


def draw_lines_and_points(image, horizontal_lines, vertical_lines, intersections):
    """
    Draw detected lines and intersection points on a copy of the image.
    """
    vis = image.copy()

    # Draw horizontal lines (blue)
    for x1, y1, x2, y2 in horizontal_lines:
        cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw vertical lines (green)
    for x1, y1, x2, y2 in vertical_lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw intersection points (red dots)
    for (x, y) in intersections:
        cv2.circle(vis, (x, y), radius=5, color=(0, 0, 255), thickness=6)

    return vis


def main(image_path):
    # Load image
    img = cv2.imread(image_path)
    img = img[500:1700, 440:1700]

    if img is None:
        print(f"Error loading image: {image_path}")
        return

    # 1. Find horizontal and vertical lines
    horizontal_lines, vertical_lines = find_lines(img)

    print(f"Found {len(horizontal_lines)} horizontal lines.")
    print(f"Found {len(vertical_lines)} vertical lines.")

    # 2. Find intersections
    intersections = get_intersections(horizontal_lines, vertical_lines)
    print(f"Found {len(intersections)} intersection points.")

    # 3. Draw lines + intersections
    vis = draw_lines_and_points(
        img, horizontal_lines, vertical_lines, intersections)

    # 4. Show with matplotlib (BGR → RGB)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(vis_rgb)
    plt.title("Horizontal/Vertical Lines and Intersections")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Change this to your image path
    main(r"./fotos/cubo6cm.jpg")
