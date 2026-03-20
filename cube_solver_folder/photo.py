import subprocess
import numpy as np
import cv2


def take_photo(filepath):

    result = subprocess.run(
        [
            "rpicam-still",
            "--nopreview",
            "--width", "1296",
            "--height", "972",
            "--timeout", "1",          # almost no delay
            "--exposure", "normal",    # skip AE settling
            "-o", "-", "--encoding", "jpg"
        ],
        capture_output=True,
        check=True
    )

    image = cv2.imdecode(np.frombuffer(
        result.stdout, np.uint8), cv2.IMREAD_COLOR)
    print("Captured image shape:", image.shape, " to ", filepath)
    cv2.imwrite(filepath, image)
