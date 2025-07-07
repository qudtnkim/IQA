import os
import glob
import numpy as np
import cv2
import csv

# Function to calculate high-frequency to low-frequency ratio with dynamic cutoff
def calculate_frequency_ratio(image, cutoff_ratio=0.018):
    """
    Calculate the high-frequency to low-frequency ratio of an image with dynamic cutoff.

    Args:
        image: Input grayscale image (2D array).
        cutoff_ratio: Ratio to determine dynamic cutoff based on image size.

    Returns:
        Ratio of high-frequency energy to low-frequency energy.
    """
    # Apply Fourier Transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)

    # Define frequency regions
    height, width = magnitude_spectrum.shape
    center = (height // 2, width // 2)

    # Calculate dynamic cutoff based on image size
    cutoff = int(min(height, width) * cutoff_ratio)

    # Low-frequency energy (center region)
    low_freq_energy = np.sum(
        magnitude_spectrum[center[0] - cutoff:center[0] + cutoff, center[1] - cutoff:center[1] + cutoff]
    )

    # Total energy minus low-frequency gives high-frequency energy
    total_energy = np.sum(magnitude_spectrum)
    high_freq_energy = total_energy - low_freq_energy

    # Avoid division by zero
    if low_freq_energy == 0:
        return 0

    # Calculate high-frequency to low-frequency ratio
    return high_freq_energy / low_freq_energy

# Function to extract triangular region with standard ROI size
def extract_triangle(image, standard_size=(1255, 1080), ratio_w=90/1255, ratio_h=100/1080):
    """
    Crop the image to remove triangular regions using standard dimensions.

    Args:
        image: Input grayscale image (2D array).
        standard_size: Standard image size to resize before cropping.
        ratio_w: Width ratio to determine crop area.
        ratio_h: Height ratio to determine crop area.

    Returns:
        Cropped image excluding triangular regions.
    """
    # Resize image to standard size
    resized_image = cv2.resize(image, standard_size)

    # Calculate crop dimensions
    height, width = resized_image.shape[:2]
    t_width = int(ratio_w * width)
    t_height = int(ratio_h * height)

    crop_width = width - 2 * t_width
    crop_height = height - 2 * t_height

    center_y, center_x = height // 2, width // 2
    start_y = max(center_y - crop_height // 2, 0)
    end_y = min(center_y + crop_height // 2, height)
    start_x = max(center_x - crop_width // 2, 0)
    end_x = min(center_x + crop_width // 2, width)

    return resized_image[start_y:end_y, start_x:end_x]

# Define CSV output file
output_csv = "250120_S_CHANNEL_Clarity_result.csv"
freq_ratios = []  # Store frequency ratios for normalization
image_names = []  # Store image names

# Get list of images
image_paths = glob.glob("../1-0.Train_DB/*.png")
total_images = len(image_paths)

# First pass: Calculate frequency ratios for all images
print(f"Processing {total_images} images...")
for idx, image_path in enumerate(image_paths, start=1):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {image_path}. Skipping.")
        continue

    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract S channel and crop
    s_channel = hsv_image[:, :, 1]
    s_triangle = extract_triangle(s_channel)

    # Calculate dynamic frequency ratio
    s_freq_ratio = calculate_frequency_ratio(s_triangle)
    freq_ratios.append(s_freq_ratio)
    image_names.append(os.path.basename(image_path))

    # Print progress every 10 images
    if idx % 10 == 0 or idx == total_images:
        print(f"[{idx}/{total_images}] Processed {os.path.basename(image_path)}")

# Normalize frequency ratios (Z-score normalization)
mean_ratio = np.mean(freq_ratios)
std_ratio = np.std(freq_ratios)
normalized_ratios = [(x - mean_ratio) / std_ratio for x in freq_ratios]

# Second pass: Write normalized results to CSV
print("Writing results to CSV...")
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name", "Normalized S Frequency Ratio"])

    for image_name, norm_ratio in zip(image_names, normalized_ratios):
        writer.writerow([image_name, norm_ratio])
print(f"Results saved to {output_csv}")
