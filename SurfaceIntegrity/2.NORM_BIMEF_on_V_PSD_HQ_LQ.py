import os
import glob
import argparse
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List

def calculate_local_fft_energy(image: np.ndarray, patch_size: Tuple[int, int] = (16, 16)) -> Tuple[float, float]:
    """
    Calculates the mean and standard deviation of FFT energy across image patches.
    This serves as a metric for the overall texture and frequency content.

    Args:
        image (np.ndarray): Input grayscale image.
        patch_size (Tuple[int, int]): The size of patches to analyze.

    Returns:
        A tuple containing the mean and standard deviation of patch energies.
    """
    h, w = image.shape
    patch_energies = []
    for i in range(0, h, patch_size[0]):
        for j in range(0, w, patch_size[1]):
            patch = image[i:i + patch_size[0], j:j + patch_size[1]]
            if patch.size > 0:
                # Energy is the sum of the magnitudes in the frequency domain
                patch_fft = np.fft.fft2(patch)
                patch_energy = np.sum(np.abs(patch_fft))
                patch_energies.append(patch_energy)
    
    if not patch_energies:
        return 0.0, 0.0
        
    return np.mean(patch_energies), np.std(patch_energies)

def apply_bimef(v_channel: np.ndarray, mu: float = 0.5, sigma: int = 5) -> np.ndarray:
    """
    Applies a simplified BIMEF algorithm to a single channel (V-channel).
    This enhances the local contrast and brightness of the image.

    Args:
        v_channel (np.ndarray): The V-channel (brightness) of an HSV image.
        mu (float): Gamma correction coefficient for the weight map.
        sigma (int): Standard deviation for the Gaussian blur to create the illumination map.

    Returns:
        The enhanced V-channel.
    """
    # Normalize to [0, 1] for processing
    img_float = v_channel.astype(np.float32) / 255.0

    # Illumination map is a blurred version of the V-channel
    illumination_map = cv2.GaussianBlur(img_float, (2 * sigma + 1, 2 * sigma + 1), sigma)
    
    # Weight map controls the enhancement level
    weight_map = np.clip(illumination_map ** mu, 0.0, 1.0)
    
    # Apply enhancement
    enhanced_float = img_float * weight_map
    
    # Normalize back to [0, 255]
    enhanced_uint8 = np.clip(enhanced_float * 255, 0, 255).astype(np.uint8)
    
    return enhanced_uint8

def crop_to_central_roi(image: np.ndarray, fixed_size: Tuple[int, int] = (512, 512)) -> np.ndarray:
    """
    Resizes an image to a fixed size and crops to a central region to remove borders.
    """
    resized = cv2.resize(image, fixed_size)
    h, w = resized.shape[:2]
    
    # These ratios define the percentage of the border to crop
    ratio_w, ratio_h = 90/1255, 100/1080
    crop_w = int(ratio_w * w)
    crop_h = int(ratio_h * h)
    
    return resized[crop_h:h - crop_h, crop_w:w - crop_w]

def determine_fidelity_level(increase_ratio: float) -> int:
    """ Maps the FFT energy increase ratio to a discrete Fidelity Level (0-5). """
    if increase_ratio <= 0:   return 0
    if increase_ratio <= 1:   return 1
    if increase_ratio <= 5:   return 2
    if increase_ratio <= 10:  return 3
    if increase_ratio <= 15:  return 4
    return 5

def save_comparison_visualization(original: np.ndarray, enhanced: np.ndarray, save_path: str):
    """ Saves a side-by-side comparison of the original and enhanced images. """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original (Cropped)")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Enhanced (BIMEF on V-Channel)")
    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()

def process_single_image(image_path: str, output_dir: str, mu: float, save_viz: bool) -> Dict[str, Any]:
    """
    Processes a single image to calculate its surface integrity score.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 1. Pre-process the image
    cropped_image = crop_to_central_roi(image)
    
    # 2. Apply BIMEF enhancement on the V-channel
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    enhanced_v = apply_bimef(v_channel, mu=mu)
    hsv[:, :, 2] = enhanced_v
    enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. Quantify texture before and after enhancement
    original_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    
    orig_fft_mean, orig_fft_std = calculate_local_fft_energy(original_gray)
    enh_fft_mean, enh_fft_std = calculate_local_fft_energy(enhanced_gray)

    # 4. Calculate the final score: the percentage increase in texture
    if orig_fft_mean == 0:
        increase_ratio = np.inf if enh_fft_mean > 0 else 0
    else:
        increase_ratio = (enh_fft_mean - orig_fft_mean) / orig_fft_mean * 100

    # 5. Classify and prepare results
    fidelity_level = determine_fidelity_level(increase_ratio)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    level_dir = os.path.join(output_dir, f"Fidelity_{fidelity_level}")
    os.makedirs(level_dir, exist_ok=True)
    
    # Save the enhanced image
    cv2.imwrite(os.path.join(level_dir, f"{base_name}_enhanced.png"), enhanced_image)
    
    # Optionally save comparison visualization
    if save_viz:
        viz_path = os.path.join(level_dir, f"{base_name}_comparison.png")
        save_comparison_visualization(cropped_image, enhanced_image, viz_path)

    return {
        "image_name": base_name,
        "fidelity_level": fidelity_level,
        "fft_increase_ratio": increase_ratio,
        "original_fft_mean": orig_fft_mean,
        "enhanced_fft_mean": enh_fft_mean,
    }

def main(args):
    image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    if not image_paths:
        print(f"[ERROR] No PNG images found in '{args.input_dir}'.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    results: List[Dict[str, Any]] = []

    print(f"--- Surface Integrity Analysis via Enhancement Potential ---")
    print(f"Found {len(image_paths)} images to process.")
    print(f"BIMEF mu parameter: {args.mu}")

    for path in image_paths:
        try:
            result = process_single_image(path, args.output_dir, args.mu, args.save_visualizations)
            results.append(result)
            print(f"Processed '{os.path.basename(path)}' -> Fidelity Level: {result['fidelity_level']}")
        except Exception as e:
            print(f"Failed to process '{os.path.basename(path)}': {e}")
    
    # Save summary CSV
    csv_path = os.path.join(args.output_dir, "surface_integrity_results.csv")
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"\n--- Analysis Complete ---")
    print(f"Results saved in '{args.output_dir}'")
    print(f"Summary CSV saved to '{csv_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantify image surface integrity based on its enhancement potential.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory containing input PNG images.")
    parser.add_argument("-o", "--output_dir", default="./surface_integrity_output", help="Directory to save results.")
    parser.add_argument("--mu", type=float, default=0.5, help="Mu parameter for the BIMEF enhancement algorithm.")
    parser.add_argument("--save_visualizations", action="store_true", help="Save side-by-side comparison images.")
    
    args = parser.parse_args()
    main(args)
