import os
import glob
import numpy as np
import cv2
import shutil

# Create folders for outputs
output_folder = "output_images"
blur_folder = os.path.join(output_folder, "blurred")
visualization_folder = os.path.join(output_folder, "visualization")

os.makedirs(blur_folder, exist_ok=True)
os.makedirs(visualization_folder, exist_ok=True)

# Function to calculate high-frequency to low-frequency ratio with dynamic cutoff
def calculate_frequency_ratio(image, cutoff_ratio=0.018):
    """
    Calculate the high-frequency to low-frequency ratio of an image with dynamic cutoff.

    Args:
        image: Input grayscale image (2D array).
        cutoff_ratio: Ratio to determine dynamic cutoff based on image size.

    Returns:
        Ratio of high-frequency energy to low-frequency energy and FFT magnitude spectrum.
    """
    # Apply Fourier Transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)  # Log for better visualization

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
        return 0, magnitude_spectrum

    return high_freq_energy / low_freq_energy, magnitude_spectrum

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

# Process all images in the current folder
threshold = 0.8  # Define a threshold for blur detection
image_ratios = []  # To store frequency ratios for normalization
image_paths = []   # To store image paths

# First pass: Calculate frequency ratios
for image_path in glob.glob("../1-0.Train_DB/*.png"):
    # Read the image in HSV format
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {image_path}. Skipping.")
        continue

    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract S channel
    s_channel = hsv_image[:, :, 1]

    # Extract the triangular ROI for S channel
    s_triangle = extract_triangle(s_channel)

    # Calculate dynamic frequency ratio and get FFT magnitude spectrum
    s_freq_ratio, fft_magnitude = calculate_frequency_ratio(s_triangle)
    image_ratios.append(s_freq_ratio)
    image_paths.append(image_path)

    # Save visualization
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    hsv_output_path = os.path.join(visualization_folder, f"{base_name}_HSV.png")
    s_channel_output_path = os.path.join(visualization_folder, f"{base_name}_S_Channel.png")
    fft_output_path = os.path.join(visualization_folder, f"{base_name}_FFT.png")

    cv2.imwrite(hsv_output_path, cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR))
    cv2.imwrite(s_channel_output_path, s_channel)
    cv2.imwrite(fft_output_path, np.uint8(255 * fft_magnitude / np.max(fft_magnitude)))

# Normalize frequency ratios (Z-score normalization)
mean_ratio = np.mean(image_ratios)
std_ratio = np.std(image_ratios)
normalized_ratios = [(x - mean_ratio) / std_ratio for x in image_ratios]

# Calculate Strong Threshold (mean + 3 * std)
strong_threshold = mean_ratio + 3 * std_ratio
print(f"Strong Threshold: {strong_threshold:.2f}")

# Second pass: Check if the image is blurred and copy it if necessary
for image_path, norm_ratio in zip(image_paths, normalized_ratios):
    if norm_ratio > threshold:  # Adjust threshold for normalized values
        blur_path_s = os.path.join(blur_folder, "S_" + os.path.basename(image_path))
        print(f"Image {image_path} is blurred in S channel (normalized ratio: {norm_ratio:.2f}). Copying to {blur_path_s}.")
        shutil.copy(image_path, blur_path_s)

# Output the Strong Threshold
print(f"Final Strong Threshold for Clear Images: {strong_threshold:.2f}")