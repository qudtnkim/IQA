import os
import glob
import csv
import argparse
import numpy as np
import cv2
from typing import Tuple, List

# --- Core Image Processing Functions ---

def calculate_hlfr(image: np.ndarray, cutoff_ratio: float = 0.02) -> float:
    """
    Calculates the High-to-Low Frequency Ratio (HLFR) for a grayscale image.

    This metric quantifies image sharpness by comparing the energy in high-frequency
    components to that in low-frequency components. A higher HLFR value
    generally corresponds to a sharper image.

    Args:
        image (np.ndarray): Input grayscale image (2D NumPy array).
        cutoff_ratio (float): A ratio of the image's smaller dimension, used to
                              define the boundary of the low-frequency region.

    Returns:
        float: The calculated HLFR score.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array.")

    # 1. Apply 2D Fast Fourier Transform (FFT) and shift the zero-frequency
    #    component to the center of the spectrum.
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)

    # 2. Define the low-frequency region based on the cutoff ratio.
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    cutoff = int(min(rows, cols) * cutoff_ratio)

    # 3. Sum the energy within the low-frequency region (a central square).
    low_freq_energy = np.sum(
        magnitude_spectrum[
            center_row - cutoff : center_row + cutoff,
            center_col - cutoff : center_col + cutoff,
        ]
    )

    # 4. High-frequency energy is the total energy minus the low-frequency energy.
    total_energy = np.sum(magnitude_spectrum)
    high_freq_energy = total_energy - low_freq_energy

    # 5. Calculate the ratio, handling potential division by zero.
    if low_freq_energy == 0:
        return np.inf  # Or a large number, as this implies all energy is high-frequency
    
    return high_freq_energy / low_freq_energy

def crop_to_central_roi(image: np.ndarray, target_size: Tuple[int, int] = (1255, 1080)) -> np.ndarray:
    """
    Resizes an image to a standard size and crops it to a central region of interest.
    
    This function is primarily used to remove potential artifacts or irrelevant
    information from the borders of an image by focusing on its central area.

    Args:
        image (np.ndarray): The input grayscale image.
        target_size (Tuple[int, int]): The standard (width, height) to which the image
                                     is resized before cropping.

    Returns:
        np.ndarray: The cropped central region of the image.
    """
    resized_image = cv2.resize(image, target_size)
    h, w = resized_image.shape

    # Define margins to crop from each side (e.g., 10% from height, 7% from width)
    crop_margin_h = int(h * 0.10)
    crop_margin_w = int(w * 0.07)
    
    start_row, end_row = crop_margin_h, h - crop_margin_h
    start_col, end_col = crop_margin_w, w - crop_margin_w
    
    return resized_image[start_row:end_row, start_col:end_col]


# --- Main Application Logic ---

def main(args):
    """
    Main function to process a directory of images:
    1. Calculates a clarity score (HLFR) for each image's S-channel.
    2. Normalizes the scores using Z-score standardization.
    3. Saves the image names, raw scores, and normalized scores to a CSV file.
    """
    image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    if not image_paths:
        print(f"Error: No PNG images found in the specified directory: '{args.input_dir}'")
        return

    print(f"Found {len(image_paths)} images. Starting clarity analysis...")
    
    results = []
    total_images = len(image_paths)

    for i, path in enumerate(image_paths):
        image_name = os.path.basename(path)
        image_bgr = cv2.imread(path)
        
        if image_bgr is None:
            print(f"Warning: Could not read image '{image_name}'. Skipping.")
            continue

        # Convert image to HSV color space and extract the Saturation (S) channel.
        # The S-channel is often effective for capturing texture and clarity.
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        s_channel = image_hsv[:, :, 1]
        
        # Crop the S-channel to the central region to avoid edge artifacts.
        s_roi = crop_to_central_roi(s_channel)
        
        # Calculate the High-to-Low Frequency Ratio.
        hlfr_score = calculate_hlfr(s_roi)
        
        results.append({"Image Name": image_name, "HLFR_Score": hlfr_score})
        
        # Print progress update
        if (i + 1) % 50 == 0 or (i + 1) == total_images:
            print(f"  Processed [{i + 1}/{total_images}] images...")

    # --- Data Normalization and Saving ---
    if not results:
        print("No images were processed successfully. Exiting.")
        return

    hlfr_scores = np.array([res["HLFR_Score"] for res in results])
    mean_score = np.mean(hlfr_scores)
    std_score = np.std(hlfr_scores)
    
    print("\nCalculating Z-scores for normalization...")
    # Z-score = (value - mean) / std_deviation
    normalized_scores = (hlfr_scores - mean_score) / std_score
    
    for i, res in enumerate(results):
        res["Normalized_HLFR_Score"] = normalized_scores[i]

    print(f"Writing results to '{args.output_csv}'...")
    with open(args.output_csv, 'w', newline='') as csvfile:
        fieldnames = ["Image Name", "HLFR_Score", "Normalized_HLFR_Score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
    print("\nAnalysis complete. Results saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyzes the clarity of images in a directory using a frequency-based "
                    "metric (HLFR) and saves the raw and normalized scores to a CSV file."
    )
    
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing the input PNG images."
    )
    parser.add_argument(
        "-o", "--output_csv",
        type=str,
        required=True,
        help="Path to save the output CSV file with clarity scores."
    )
    
    args = parser.parse_args()
    main(args)
