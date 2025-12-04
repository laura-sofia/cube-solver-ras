import cv2
import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def group_hsv_kmeans(hsv_array, n_clusters=6, random_state=0):
    """
    Parameters
    ----------
    hsv_array : array-like
        Array of shape (6, 9, 3) (or any (H, W, 3)) with HSV values.
    n_clusters : int
        Number of color groups (default: 6).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    labels_2d : np.ndarray
        Array of shape (6, 9) (or (H, W)) with group indices 0..n_clusters-1.
    """
    hsv_array = np.asarray(hsv_array)
    
    # Save the spatial shape (6, 9)
    h, w = hsv_array.shape[:2]
    
    # Flatten to (N, 3) where N = 6*9
    pixels = hsv_array.reshape(-1, hsv_array.shape[-1])
    
    # Run k-means in HSV space
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(pixels)  # shape (N,)
    
    # Reshape labels back to (6, 9)
    labels_2d = labels.reshape(h, w)
    return labels_2d

def plot_hsv_clusters(hsv_grid, labels_grid):
    """
    Visualize HSV colors (top row) and their k-means cluster IDs (bottom row).

    Parameters
    ----------
    hsv_grid : (H, W, 3) array of HSV values
    labels_grid : (H, W) array of cluster indices
    """
    H, W = labels_grid.shape

    # Flatten grids into 1D sequences
    hsv_flat = hsv_grid.reshape(-1, 3)
    labels_flat = labels_grid.reshape(-1)

    # Convert HSV (OpenCV format) -> RGB for plotting
    hsv_uint8 = hsv_flat.astype(np.uint8).reshape(1, -1, 3)
    rgb_flat = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2RGB).reshape(-1, 3) / 255.0

    N = hsv_flat.shape[0]

    # Prepare the figure
    plt.figure(figsize=(12, 2))

    # --- TOP LINE: original colors ---
    for i in range(N):
        plt.scatter(i, 1, color=rgb_flat[i], s=200)

    # --- BOTTOM LINE: cluster labels ---
    # Each label is plotted as a grayscale or category color
    cmap = plt.get_cmap("tab10")  # distinct colors for clusters
    for i in range(N):
        plt.scatter(i, 0, color=cmap(labels_flat[i] % 10), s=200)
        plt.text(i, -0.1, str(labels_flat[i]), ha='center', va='top', fontsize=8)

    plt.ylim(0, 0.1)
    plt.yticks([0, 1], ["Cluster", "Color"])
    plt.xticks([])

    plt.title("HSV Colors (top) and K-means Clusters (bottom)")
    plt.grid(False)
    plt.show()

import numpy as np

def generate_noisy_hsv_grid(n_colors, rows=6, cols=9,
                            noise_std=(5, 20, 20), seed=None):
    """
    Generate a HSV grid where each cell is based on one of n_colors base HSVs,
    with added noise per channel.

    Parameters
    ----------
    n_colors : int
        Number of distinct base colors (clusters).
    rows, cols : int
        Size of the grid (default 6x9).
    noise_std : tuple of float
        Standard deviation of Gaussian noise for (H, S, V).
        In OpenCV HSV ranges: H∈[0,179], S∈[0,255], V∈[0,255].
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    hsv_grid : np.ndarray
        Array of shape (rows, cols, 3) with noisy HSV values (dtype uint8).
    labels_grid : np.ndarray
        Array of shape (rows, cols) with base color indices (0..n_colors-1).
    base_colors : np.ndarray
        Array of shape (n_colors, 3) with the original base HSV colors.
    """
    rng = np.random.default_rng(seed)

    # Generate base HSV colors (a bit saturated & bright so they’re visible)
    base_H = rng.integers(0, 180, size=n_colors)
    base_S = rng.integers(0, 256, size=n_colors)
    base_V = rng.integers(120, 256, size=n_colors)
    base_colors = np.stack([base_H, base_S, base_V], axis=1).astype(np.int16)

    # Total cells
    N = rows * cols

    # Assign each cell to a base color, roughly uniformly, ensuring each appears at least once
    if n_colors > N:
        raise ValueError("n_colors cannot be greater than rows*cols")

    labels_flat = rng.integers(0, n_colors, size=N)

    # Ensure every color is used at least once
    for k in range(n_colors):
        labels_flat[k] = k
    rng.shuffle(labels_flat)

    labels_grid = labels_flat.reshape(rows, cols)

    # Create noisy HSV grid
    hsv_grid = np.zeros((rows, cols, 3), dtype=np.uint8)

    h_noise_std, s_noise_std, v_noise_std = noise_std

    for idx, base_idx in enumerate(labels_flat):
        base = base_colors[base_idx]  # int16

        # Add Gaussian noise per channel
        h = base[0] + rng.normal(0, h_noise_std)
        s = base[1] + rng.normal(0, s_noise_std)
        v = base[2] + rng.normal(0, v_noise_std)

        # Clip to valid OpenCV HSV ranges
        h = np.clip(h, 0, 179)
        s = np.clip(s, 0, 255)
        v = np.clip(v, 0, 255)

        r = idx // cols
        c = idx % cols
        hsv_grid[r, c, :] = [int(h), int(s), int(v)]

    return hsv_grid, labels_grid, base_colors

def Test():
    hsv_grid, true_labels, base_colors = generate_noisy_hsv_grid(
        n_colors=6, rows=6, cols=9, noise_std=(5, 5, 5), seed=None
    )

    labels = group_hsv_kmeans(hsv_grid)
    print(f"Label size: {labels.shape}")
    plot_hsv_clusters(hsv_grid, labels)


if __name__ == "__main__":
    Test()