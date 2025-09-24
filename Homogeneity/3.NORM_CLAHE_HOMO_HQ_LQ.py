import os
import glob
import argparse
from typing import Tuple, Dict, Any, List

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.ndimage import sobel

# --- 1. Pre-processing Functions ---

def crop_to_central_roi(image: np.ndarray, width_ratio: float = 90/1255, height_ratio: float = 100/1080) -> np.ndarray:
    """
    Crops the image to a central region to remove potential border artifacts.

    Args:
        image (np.ndarray): The input image (BGR or grayscale).
        width_ratio (float): The ratio of the total width to be cropped from each side.
        height_ratio (float): The ratio of the total height to be cropped from each side.

    Returns:
        np.ndarray: The cropped central region of the image.
    """
    h, w = image.shape[:2]
    crop_w = int(width_ratio * w)
    crop_h = int(height_ratio * h)

    center_y, center_x = h // 2, w // 2
    
    # Calculate crop box dimensions
    box_w = w - 2 * crop_w
    box_h = h - 2 * crop_h

    start_y = max(center_y - box_h // 2, 0)
    end_y = min(center_y + box_h // 2, h)
    start_x = max(center_x - box_w // 2, 0)
    end_x = min(center_x + box_w // 2, w)

    return image[start_y:end_y, start_x:end_x]

# --- 2. Feature Extraction Functions ---

def detect_pattern_irregularity(image_v_channel: np.ndarray, patch_ratio: float, threshold_factor: float) -> np.ndarray:
    """
    Detects pattern irregularities in the V-channel (brightness) of an image
    using a patch-based FFT analysis.

    Args:
        image_v_channel (np.ndarray): The V-channel of an HSV image.
        patch_ratio (float): The ratio to determine the patch size relative to image dimensions.
        threshold_factor (float): A multiplier for the mean magnitude to set an adaptive threshold.

    Returns:
        np.ndarray: A binary mask where high values indicate irregular regions.
    """
    h, w = image_v_channel.shape
    patch_size = max(4, int(min(h, w) * patch_ratio))
    irregularity_mask = np.zeros_like(image_v_channel, dtype=np.uint8)

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image_v_channel[i:i+patch_size, j:j+patch_size]
            if patch.size == 0:
                continue

            # Analyze the frequency spectrum of the patch
            f_transform = np.fft.fft2(patch)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)

            # Flag regions with high-frequency content above an adaptive threshold
            adaptive_threshold = threshold_factor * np.mean(magnitude_spectrum)
            if np.any(magnitude_spectrum > adaptive_threshold):
                irregularity_mask[i:i+patch_size, j:j+patch_size] = 255
                
    return irregularity_mask

def compute_psd_energy(image: np.ndarray, band: Tuple[float, float] = (0.01, 0.49), fs: float = 1.0) -> float:
    """
    Computes the Power Spectral Density (PSD) energy within a specified frequency band.
    Higher energy often correlates with richer texture and detail.

    Args:
        image (np.ndarray): The input BGR image.
        band (Tuple[float, float]): The frequency band of interest.
        fs (float): The sampling frequency.

    Returns:
        float: The integrated PSD energy within the band.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    signal = gray.flatten().astype(np.float64)
    signal -= np.mean(signal)  # Detrend the signal

    freqs, power = welch(signal, fs=fs, nperseg=max(1, int(len(signal) * 0.2)), scaling='density')
    
    # Integrate power within the specified frequency band
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    band_energy = np.trapz(power[band_mask], freqs[band_mask])

    return band_energy

def calculate_mask_complexity(mask: np.ndarray, patch_ratio: float) -> float:
    """
    Quantifies the complexity of the irregularity mask. A simple, uniform mask
    will have low complexity, while a chaotic one will have high complexity.

    Args:
        mask (np.ndarray): The binary irregularity mask.
        patch_ratio (float): Ratio to determine patch size for analysis.

    Returns:
        float: The calculated complexity score (variance of edge densities).
    """
    h, w = mask.shape
    patch_size = max(4, int(min(h, w) * patch_ratio))
    
    # Use Sobel operator to find edges within the mask
    sobel_edges = sobel(mask.astype(float))
    edge_densities = []

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch_edges = sobel_edges[i:i+patch_size, j:j+patch_size]
            if patch_edges.size > 0:
                density = np.sum(patch_edges > 0) / patch_edges.size
                edge_densities.append(density)

    if not edge_densities or np.all(np.array(edge_densities) == 0):
        return 0.0

    # Complexity is the variance of these normalized densities
    complexity = np.var(edge_densities)
    return complexity

# --- 3. Image Enhancement and Scoring ---

def enhance_irregular_regions(image: np.ndarray, irregularity_mask: np.ndarray) -> np.ndarray:
    """
    Selectively enhances contrast in irregular regions using CLAHE.

    Args:
        image (np.ndarray): The input BGR image.
        irregularity_mask (np.ndarray): The mask identifying regions to enhance.

    Returns:
        np.ndarray: The selectively enhanced image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    v_clahe = clahe.apply(v_channel)

    # Apply enhancement only where the mask is active
    v_enhanced = np.where(irregularity_mask == 255, v_clahe, v_channel)
    
    hsv[:, :, 2] = v_enhanced
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def calculate_homogeneity_score(complexity: float, psd_energy: float, weights: Tuple[float, float]) -> float:
    """
    Calculates a final Homogeneity score based on complexity and PSD energy.
    The score is designed so that low complexity and high PSD energy result in a higher score.

    Args:
        complexity (float): The mask complexity score. Lower is better.
        psd_energy (float): The PSD energy score. Higher is better.
        weights (Tuple[float, float]): Weights for (complexity, psd_energy).

    Returns:
        float: The final combined homogeneity score.
    """
    w_complexity, w_psd = weights
    # The score is inversely proportional to complexity and directly proportional to PSD energy.
    score = (1 / (complexity + 1e-7)) * w_complexity + psd_energy * w_psd
    return score

# --- 4. Classification and Visualization ---

def classify_and_save(original_img: np.ndarray, enhanced_img: np.ndarray, mask: np.ndarray, 
                      image_path: str, complexity: float, psd_energy: float, weights: Tuple[float, float],
                      output_dir: str) -> Dict[str, Any]:
    """
    Classifies the image into a fidelity level based on its homogeneity score
    and saves a comparison plot.

    Args:
        All inputs are self-explanatory.

    Returns:
        Dict[str, Any]: A dictionary containing the analysis results for the image.
    """
    homogeneity_score = calculate_homogeneity_score(complexity, psd_energy, weights)

    # These thresholds define the classification boundaries for fidelity levels.
    if homogeneity_score > 4:    level = 5  # Clear
    elif 3 < homogeneity_score <= 4: level = 4  # Very High
    elif 2 < homogeneity_score <= 3: level = 3  # High
    elif 1 < homogeneity_score <= 2: level = 2  # Moderate
    else:                        level = 1  # Low
    
    level_folder = os.path.join(output_dir, f"Fidelity_Level_{level}")
    os.makedirs(level_folder, exist_ok=True)

    img_name = os.path.basename(image_path)
    save_path = os.path.join(level_folder, f"comparison_{img_name}")

    # Generate and save a comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Cropped Image")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Selectively Enhanced")
    axes[1].axis("off")
    axes[2].imshow(mask, cmap="gray")
    axes[2].set_title(f"Irregularity Mask\nHomogeneity Score: {homogeneity_score:.2f}")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

    return {
        "image_name": img_name,
        "homogeneity_score": homogeneity_score,
        "complexity_score": complexity,
        "psd_energy_score": psd_energy,
        "fidelity_level": level,
    }

# --- Main Processing Pipeline ---

def process_image_folder(args: argparse.Namespace):
    """
    Main pipeline to process all images in a folder.
    """
    image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    if not image_paths:
        print(f"[ERROR] No PNG images found in '{args.input_dir}'. Please check the path.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []

    print(f"--- Starting Homogeneity Analysis ---")
    print(f"Input Directory: {args.input_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Parameters: patch_ratio={args.patch_ratio}, threshold_factor={args.threshold_factor}")
    print(f"Found {len(image_paths)} images to process.")

    for idx, path in enumerate(image_paths, 1):
        print(f"\n[INFO] Processing image {idx}/{len(image_paths)}: {os.path.basename(path)}")
        
        image = cv2.imread(path)
        if image is None:
            print(f"[WARNING] Could not read image, skipping.")
            continue

        # 1. Pre-process by cropping
        cropped_image = crop_to_central_roi(image)
        
        # 2. Extract features
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        
        irregularity_mask = detect_pattern_irregularity(v_channel, args.patch_ratio, args.threshold_factor)
        complexity = calculate_mask_complexity(irregularity_mask, args.patch_ratio)
        psd_energy = compute_psd_energy(cropped_image)
        
        # 3. Enhance image based on features
        enhanced_image = enhance_irregular_regions(cropped_image, irregularity_mask)
        
        # 4. Classify, save results, and collect data
        result = classify_and_save(
            cropped_image, enhanced_image, irregularity_mask, path,
            complexity, psd_energy, tuple(args.weights), args.output_dir
        )
        results.append(result)

    # Save final results to a single CSV file
    output_csv_path = os.path.join(args.output_dir, "homogeneity_analysis_results.csv")
    pd.DataFrame(results).to_csv(output_csv_path, index=False)
    print(f"\n--- Analysis Complete ---")
    print(f"All results and plots saved in '{args.output_dir}'")
    print(f"Summary CSV saved to '{output_csv_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A pipeline to quantify image homogeneity/fidelity based on pattern "
                    "irregularity, complexity, and power spectral density."
    )
    
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Path to the directory containing input PNG images.")
    parser.add_argument("-o", "--output_dir", type=str, default="./homogeneity_output", help="Directory to save output plots and the final CSV file.")
    
    # Key parameters for the algorithm
    parser.add_argument("--patch_ratio", type=float, default=0.0383, help="Ratio for determining the patch size in irregularity detection.")
    parser.add_argument("--threshold_factor", type=float, default=1.5, help="Multiplier for the adaptive FFT threshold in irregularity detection.")
    parser.add_argument("--weights", type=float, nargs=2, default=[0.8, 0.2], help="Weights for (1/complexity, psd_energy) in the final score. Provide two numbers.")

    args = parser.parse_args()
    process_image_folder(args)
