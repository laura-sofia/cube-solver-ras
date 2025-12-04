import cv2
import numpy as np
import matplotlib.pyplot as plt


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def get_square_hsv(square_img):
    """Returns average HSV of a square."""
    hsv_square = cv2.cvtColor(square_img, cv2.COLOR_BGR2HSV)
    h, s, v = np.mean(hsv_square, axis=(0, 1)).astype(int)
    return (h, s, v)


def getAverageColor(img, p, size):

    size = int(size/2)
    p.x = int(p.x)
    p.y = int(p.y)

    square_segment = img[p.y - size:p.y + size, p.x - size:p.x + size]
    return get_square_hsv(square_segment)


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


def drawCircle(vis, x, y):
    cv2.circle(vis, (int(x), int(y)), radius=5,
               color=(0, 255, 255), thickness=6)


def getMiddles(corner1, corner2):

    distX = corner2.x - corner1.x
    distY = corner2.y - corner1.y
    dist = (distX + distY) / 2

    mDist = dist/6
    l = []

    for i in range(3):
        for j in range(3):
            l.append(Point(corner1.x + i * mDist*2 + mDist,
                           corner1.y + j * mDist*2 + mDist))

    return l


def main(image_path):
    # Load image
    img = cv2.imread(image_path)
    # img = img[500:1700, 440:1700]
    img = img[100:700, 200:850]

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

    x1, y1 = 4000, 4000
    for pair in intersections:
        x1 = min(x1, pair[0])
        y1 = min(y1, pair[1])

    x2, y2 = 0, 0
    for pair in intersections:
        x2 = max(x2, pair[0])
        y2 = max(y2, pair[1])

    # 3. Draw lines + intersections
    vis = draw_lines_and_points(
        img, horizontal_lines, vertical_lines, intersections)

    drawCircle(vis, x1, y1)
    drawCircle(vis, x2, y2)
    middles = getMiddles(Point(x1, y1), Point(x2, y2))

    cubesize = ((x2 + x1)/2 + (y2 - y1)/2)/2

    colors = []

    for point in middles:

        tpl = getAverageColor(img, point, cubesize/9)
        colors.append(tpl)
        hsv_array = np.array([[tpl]], dtype=np.uint8)
        bgr_array = cv2.cvtColor(hsv_array, cv2.COLOR_HSV2BGR)[0, 0]
        rgb_tuple = (int(bgr_array[0]), int(bgr_array[1]), int(bgr_array[2]))
        cv2.circle(vis, (int(point.x), int(point.y)), radius=24,
                   color=rgb_tuple, thickness=20)
        drawCircle(vis, point.x, point.y)

    # 4. Show with matplotlib (BGR → RGB)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(vis_rgb)
    plt.title("Horizontal/Vertical Lines and Intersections")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    # Change this to your image path
    main(r"./fotos/cubo26cm.jpg")
