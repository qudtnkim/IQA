import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy.signal import welch
from scipy.ndimage import sobel

# Triangle 영역 제거 함수
def extract_triangle(image, ratio_w=90 / 1255, ratio_h=100 / 1080):
    """
    Triangular 영역을 제거한 크롭된 이미지를 반환합니다.
    """
    height, width = image.shape[:2]
    t_width = int(ratio_w * width)
    t_height = int(ratio_h * height)

    crop_width = width - 2 * t_width
    crop_height = height - 2 * t_height

    center_y, center_x = height // 2, width // 2
    start_y = max(center_y - crop_height // 2, 0)
    end_y = min(center_y + crop_height // 2, height)
    start_x = max(center_x - crop_width // 2, 0)
    end_x = min(center_x + crop_width // 2, width)

    return image[start_y:end_y, start_x:end_x]

# FFT 기반 불규칙성 탐지
def detect_pattern_irregularity(image, patch_ratio=0.0383, threshold_factor=1.5):
    """
    FFT 기반으로 이미지 불규칙성을 탐지하고 마스크를 생성합니다.
    """
    h, w = image.shape
    patch_size = max(4, int(min(h, w) * patch_ratio))
    mask = np.zeros_like(image, dtype=np.uint8)

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            if patch.size == 0:
                continue

            f_transform = np.fft.fft2(patch)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)

            adaptive_threshold = threshold_factor * np.mean(magnitude_spectrum)
            irregular_patch = (magnitude_spectrum > adaptive_threshold).astype(np.uint8) * 255
            mask[i:i+patch_size, j:j+patch_size] = irregular_patch

    return mask

# PSD Energy 계산
def compute_psd(image, band=(0.01, 0.49), fs=1.0, percent_nperseg=0.2):
    """
    PSD 에너지 계산.

    Args:
        image: 입력 이미지.
        band: PSD 계산 시 사용할 주파수 대역.
        fs: 샘플링 주파수.
        percent_nperseg: FFT 윈도우 길이 비율.

    Returns:
        float: 주파수 대역 내 PSD 에너지.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flattened_signal = gray.flatten().astype(np.float64)
    flattened_signal -= np.mean(flattened_signal)

    nperseg = max(1, int(len(flattened_signal) * percent_nperseg))
    freqs, power = welch(flattened_signal, fs=fs, nperseg=nperseg, scaling='density')

    # 대역 에너지 계산
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    band_energy = np.trapz(power[band_mask], freqs[band_mask])

    return band_energy

# CLAHE 기반 밝기 개선
def dynamic_clahe_enhancement(image, irregularity_mask, clip_limit=2.0, tile_grid_size=(16, 16)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v_clahe = clahe.apply(v)

    v_enhanced = np.where(irregularity_mask == 255, v_clahe, v)
    hsv[:, :, 2] = v_enhanced
    enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return enhanced_image

# 복잡성 및 변화도 계산
def calculate_irregularity_complexity(mask, patch_ratio=0.0383):
    h, w = mask.shape
    patch_size = max(4, int(min(h, w) * patch_ratio))  # Adaptive patch size
    edge_density_list = []

    # Sobel 연산으로 에지 계산
    sobel_edges = sobel(mask)

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = mask[i:i+patch_size, j:j+patch_size]
            if patch.size == 0:
                continue

            patch_edges = sobel_edges[i:i+patch_size, j:j+patch_size]
            edge_density = np.sum(patch_edges > 0) / patch.size
            edge_density_list.append(edge_density)
 
    # 정규화된 복잡성 계산
    edge_density_list = (edge_density_list - np.min(edge_density_list)) / (np.max(edge_density_list) - np.min(edge_density_list) + 1e-7)
    complexity = np.var(edge_density_list)  # 복잡성

    return complexity

def calculate_homogeneity_score(complexity, psd_energy, weights=(0.8, 0.2)):
    """
    Homogeneity 점수를 계산합니다.
    - complexity: 복잡성 (낮을수록 고품질).
    - psd_energy: PSD 에너지 (높을수록 고품질).
    - weights: (complexity, psd_energy)에 대한 가중치.
    """
    w1, w2 = weights
    epsilon = 1e-7  # Complexity가 0에 가까운 경우를 대비한 작은 값

    score = (1 / (complexity + epsilon)) * w1 + psd_energy * w2
    return score

# Homogeneity 등급 결정 및 폴더 저장 
# Homogeneity 등급 결정 및 폴더 저장
def classify_image(original, enhanced, mask, image_path, complexity, psd_energy, output_folder):
    homogeneity_score = calculate_homogeneity_score(complexity, psd_energy)

    if homogeneity_score > 4:
        level = 5  # Clear Image
    elif 3 < homogeneity_score <= 4:
        level = 4  # Very High Quality
    elif 2 < homogeneity_score <= 3:
        level = 3  # High Quality
    elif 1 < homogeneity_score <= 2:
        level = 2  # Moderate Quality
    else:
        level = 1  # Low Quality

    level_folder = os.path.join(output_folder, f"Fidelity_Level_{level}")
    os.makedirs(level_folder, exist_ok=True)

    img_name = os.path.basename(image_path)
    save_path = os.path.join(level_folder, f"comparison_{img_name}")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Enhanced")
    plt.imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Irregularity Mask\nHomogeneity: {homogeneity_score:.2f}")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.savefig(save_path)
    plt.close()

    return {
        "img_name": img_name,
        "homogeneity": homogeneity_score,
        "complexity": complexity,
        "psd_energy": psd_energy,
    }


# 이미지 처리
# 이미지 처리
def process_images(folder_path, output_csv, patch_ratio=0.0383, threshold_factor=1.5):
    """
    폴더 내 모든 이미지를 처리하여 Homogeneity를 계산합니다.

    Args:
        folder_path: 이미지 폴더 경로.
        output_csv: 결과 CSV 파일 경로.
        patch_ratio: 패치 크기 비율.
        threshold_factor: FFT 임계값 비율.
    """
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))
    results = []
    output_folder = "./output_images"  # 저장 폴더 설정

    for idx, image_path in enumerate(image_paths, 1):
        print(f"[INFO] Processing image {idx}/{len(image_paths)}: {image_path}")

        # 이미지 읽기
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] Unable to read image: {image_path}. Skipping.")
            continue

        # Triangle 영역 제거
        cropped_image = extract_triangle(image)
        hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        # 불규칙성 탐지 및 복잡성 계산
        irregularity_mask = detect_pattern_irregularity(v_channel, patch_ratio, threshold_factor)
        complexity = calculate_irregularity_complexity(irregularity_mask, patch_ratio)

        # PSD 에너지 계산
        psd_energy = compute_psd(cropped_image)

        # CLAHE 적용
        enhanced_image = dynamic_clahe_enhancement(cropped_image, irregularity_mask)

        # Homogeneity 계산 및 결과 저장
        result = classify_image(
            cropped_image,
            enhanced_image,
            irregularity_mask,
            image_path,
            complexity,
            psd_energy,
            output_folder,
        )
        results.append(result)

    # 결과 CSV 저장
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"[INFO] Results saved to {output_csv}")


if __name__ == "__main__":
    folder_path = "../1-0.Train_DB/"
    output_csv = "./250121_NORM_Homogenity_results.csv"
    process_images(folder_path, output_csv)
 
