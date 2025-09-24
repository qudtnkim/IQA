import os
import glob
import shutil
import argparse
import numpy as np
import cv2
from typing import Tuple, List

# --- Core Image Processing Functions ---

def calculate_hflr(image: np.ndarray, cutoff_ratio: float = 0.02) -> Tuple[float, np.ndarray]:
    """
    Calculates the High-to-Low Frequency Ratio (HLFR) of a grayscale image.

    This ratio serves as a metric for image sharpness. A lower ratio typically
    indicates a blurrier image, as most of the energy is concentrated in the
    low-frequency components.

    Args:
        image (np.ndarray): Input grayscale image as a 2D NumPy array.
        cutoff_ratio (float): The ratio of the image's smaller dimension used to define
                              the radius of the low-frequency region.

    Returns:
        Tuple[float, np.ndarray]: A tuple containing:
            - The calculated HLFR.
            - The log-magnitude spectrum for visualization.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    # 1. Apply 2D Fourier Transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1) # Add 1 to avoid log(0)

    # 2. Define low-frequency and high-frequency regions
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2

    # The cutoff radius is dynamically calculated based on image size
    cutoff = int(min(rows, cols) * cutoff_ratio)
    
    # 3. Calculate energy in the low-frequency region (a square mask)
    mask = np.zeros_like(image, dtype=bool)
    mask[center_row - cutoff : center_row + cutoff, center_col - cutoff : center_col + cutoff] = True
    
    low_freq_energy = np.sum(magnitude_spectrum[mask])
    total_energy = np.sum(magnitude_spectrum)
    high_freq_energy = total_energy - low_freq_energy

    # 4. Compute the ratio
    if low_freq_energy == 0:
        return np.inf, magnitude_spectrum # Avoid division by zero
    
    hlfr = high_freq_energy / low_freq_energy
    return hlfr, magnitude_spectrum


def crop_to_central_roi(image: np.ndarray, target_size: Tuple[int, int] = (1255, 1080)) -> np.ndarray:
    """
    Resizes an image to a standard size and crops it to a central region of interest.
    
    This function is designed to remove potentially noisy or irrelevant border regions,
    such as triangular artifacts from specific imaging devices.

    Args:
        image (np.ndarray): Input grayscale image.
        target_size (Tuple[int, int]): The standard (width, height) to resize to before cropping.

    Returns:
        np.ndarray: The cropped central region of the image.
    """
    # Resize to a standardized dimension for consistent ROI extraction
    resized_image = cv2.resize(image, target_size)
    h, w = resized_image.shape

    # Define crop margins (e.g., crop 10% from each side)
    crop_margin_h, crop_margin_w = int(h * 0.1), int(w * 0.1)
    
    start_row, end_row = crop_margin_h, h - crop_margin_h
    start_col, end_col = crop_margin_w, w - crop_margin_w
    
    return resized_image[start_row:end_row, start_col:end_col]


# --- Main Application Logic ---

def main(args):
    """
    Main function to analyze a directory of images, classify them as clear or
    blurry based on frequency analysis, and save the results.
    """
    # --- 1. Setup Environment ---
    print("Setting up output directories...")
    os.makedirs(args.blur_dir, exist_ok=True)
    os.makedirs(args.viz_dir, exist_ok=True)
    
    image_paths = glob.glob(os.path.join(args.input_dir, "*.png"))
    if not image_paths:
        print(f"Error: No PNG images found in '{args.input_dir}'.")
        return

    print(f"Found {len(image_paths)} images to process.")
    
    # --- 2. First Pass: Analyze all images and collect HLFR scores ---
    hlfr_scores = []
    valid_paths = []
    
    print("Analyzing image clarity (Pass 1/2)...")
    for path in image_paths:
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            print(f"Warning: Could not read {path}. Skipping.")
            continue
        
        # Convert to HSV and extract the Saturation (S) channel
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        s_channel = image_hsv[:, :, 1]
        
        # Crop to the central region to remove artifacts
        s_roi = crop_to_central_roi(s_channel)
        
        # Calculate the clarity score (HLFR)
        hlfr, fft_spectrum = calculate_hflr(s_roi)
        
        hlfr_scores.append(hlfr)
        valid_paths.append(path)

        # Save visualization artifacts
        base_name = os.path.splitext(os.path.basename(path))[0]
        fft_normalized = cv2.normalize(fft_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(os.path.join(args.viz_dir, f"{base_name}_FFT_Spectrum.png"), fft_normalized)
        cv2.imwrite(os.path.join(args.viz_dir, f"{base_name}_S_Channel_ROI.png"), s_roi)

    # --- 3. Calculate Statistical Threshold ---
    hlfr_scores = np.array(hlfr_scores)
    mean_hlfr = np.mean(hlfr_scores)
    std_hlfr = np.std(hlfr_scores)
    
    # Blurry images have low HLFR. We define the threshold to catch outliers on the low end.
    # A value of (mean - 2 * std) is a common choice to identify the bottom ~2.5% of data.
    blur_threshold = mean_hlfr - args.sigma_multiplier * std_hlfr

    print("\n" + "="*30)
    print("  Clarity Analysis Results")
    print("="*30)
    print(f"  Processed Images: {len(hlfr_scores)}")
    print(f"  Mean HLFR: {mean_hlfr:.4f}")
    print(f"  Std. Dev. HLFR: {std_hlfr:.4f}")
    print(f"  Blur Threshold (mean - {args.sigma_multiplier}σ): {blur_threshold:.4f}")
    print("="*30 + "\n")

    # --- 4. Second Pass: Classify and copy blurry images ---
    blur_count = 0
    print("Classifying images and copying blurry ones (Pass 2/2)...")
    for path, score in zip(valid_paths, hlfr_scores):
        if score < blur_threshold:
            blur_count += 1
            dest_path = os.path.join(args.blur_dir, os.path.basename(path))
            shutil.copy(path, dest_path)
            print(f"  -> BLURRY: '{os.path.basename(path)}' (Score: {score:.4f})")

    print("\n--- Analysis Complete ---")
    print(f"Total images classified as blurry: {blur_count}")
    print(f"Blurred images have been copied to: '{args.blur_dir}'")
    print(f"Visualizations saved in: '{args.viz_dir}'")


if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Analyze image clarity using High-to-Low Frequency Ratio (HLFR) "
                    "and separate blurry images based on a statistical threshold."
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing input PNG images."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="clarity_analysis_output",
        help="Path to the main output directory."
    )
    parser.add_argument(
        "--sigma_multiplier",
        type=float,
        default=2.0,
        help="Multiplier for standard deviation to set the blur threshold (e.g., 2.0 for mean - 2*sigma)."
    )
    
    args = parser.parse_args()
    
    # Define specific output subdirectories
    args.blur_dir = os.path.join(args.output_dir, "blurry_images")
    args.viz_dir = os.path.join(args.output_dir, "visualizations")

    main(args)
