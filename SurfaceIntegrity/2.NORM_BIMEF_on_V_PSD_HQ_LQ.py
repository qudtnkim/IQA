import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.integrate import trapz
import glob
import os
import pandas as pd
import random

def local_fft_normalization(image, patch_size=(16,16)):
    h, w = image.shape
    patches = []
    for i in range(0, h, patch_size[0]):
        for j in range(0, w, patch_size[1]):
            patch = image[i:i+patch_size[0], j:j+patch_size[1]]
            patch_fft = np.fft.fft2(patch)
            patch_energy = np.sum(np.abs(patch_fft))
            patches.append(patch_energy)
    return np.mean(patches), np.std(patches)


def bimef(image, mu, a=-0.3293, b=1.1258, lambda_=0.5, sigma=5): 
    """
    BIMEF 알고리즘 적용
    #mu: weight map의 gamma coefficient, 조명 맵의 영향을 제어
    #sigma: Gaussian smoothing에서의 표준 편차, 넓은 영역 평균 계산
    """
    # 단일 채널인지 확인
    if len(image.shape) == 2:  # 단일 채널 (grayscale or V channel)
        is_single_channel = True
        image = np.expand_dims(image, axis=-1)  # 채널 차원 추가
    else:
        is_single_channel = False

    # Scene illumination map 계산
    t_b = np.max(image, axis=2) if not is_single_channel else image[:, :, 0]
    t_b_smooth = cv2.GaussianBlur(t_b, (sigma * 2 + 1, sigma * 2 + 1), sigma)

    # Weight map 생성
    weight_map = np.clip(t_b_smooth ** mu, 0, 1)

    # 가중치 맵을 원본 이미지에 적용
    enhanced_image = image * weight_map[..., np.newaxis]

    # Normalize to [0, 255]
    enhanced_image = (enhanced_image - enhanced_image.min()) / (enhanced_image.max() - enhanced_image.min()) * 255
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

    # 단일 채널 이미지라면 다시 차원 축소
    if is_single_channel:
        enhanced_image = enhanced_image[:, :, 0]

    return enhanced_image


def extract_triangle(image, fixed_size=(512,512), ratio_w=90/1255, ratio_h=100/1080):
    """
    비율에 따라 삼각형 영역 제거 (크기 고정 후 작동)
    - fixed_size: 이미지 크기를 고정된 크기로 조정
    """
    # Resize image to fixed size
    resized_image = cv2.resize(image, fixed_size)

    # Triangle 영역 제거
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

    cropped_image = resized_image[start_y:end_y, start_x:end_x]
    return cropped_image

def save_visualization_and_fft(cropped_image, enhanced_image, chosen_folder, base_name, label_str):
    """
    FFT 시각화 및 비교 이미지를 저장.
    """
    if random.random() <=1:  # 10% 확률로 실행
        visualization_path = os.path.join(chosen_folder, f"{base_name}_visualization.png")
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Cropped Image")
        plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title("Enhanced Image (BIMEF on V)")
        plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.savefig(visualization_path)
        plt.close()
        print(f"[{label_str}] Visualization saved: {visualization_path}")
    else:
        print(f"[{label_str}] Visualization not saved (10% condition not met).")

def process_image(image_path, output_folder, mu, freq_range=(0.01, 0.49), results=None): 
    """
    BIMEF 적용 후 PSD 넓이 비교 및 Local FFT Normalization 계산.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # (1) 삼각형 영역 제거
    cropped_image = extract_triangle(image)

    # (2) V 채널 분리
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:, :, 2]

    # (3) BIMEF 적용 (V 채널)
    enhanced_v = bimef(v_channel, mu=mu)

    # (4) V 채널 복원 후 HSV -> BGR 변환
    hsv_image[:, :, 2] = enhanced_v
    enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    # (6) Local FFT Normalization 계산
    cropped_fft_mean, cropped_fft_std = local_fft_normalization(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY))
    enhanced_fft_mean, enhanced_fft_std = local_fft_normalization(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY))

    # 새로운 방식
    fft_increase_ratio = (enhanced_fft_mean - cropped_fft_mean) / cropped_fft_mean * 100
    #print(f"[INFO] Local FFT Mean Ratio => Increase Ratio: {increase_ratio:.4f}")

    # (8) Fidelity 단계 기준 설정
    base_name = os.path.basename(image_path).split(".")[0]
    fidelity_level = determine_fidelity_level(fft_increase_ratio)

    # Fidelity에 따라 폴더 생성
    chosen_folder = os.path.join(output_folder, f"Fidelity_{fidelity_level}")
    os.makedirs(chosen_folder, exist_ok=True)

    # 결과 이미지 저장
    enhanced_image_path = os.path.join(chosen_folder, f"{base_name}.png")
    cv2.imwrite(enhanced_image_path, enhanced_image)

    if results is not None:
        results.append({
            "image_name": base_name,
            "fidelity_level": fidelity_level,
            "fft_increase_ratio": fft_increase_ratio,
            "cropped_fft_mean": cropped_fft_mean,
            "cropped_fft_std": cropped_fft_std,
            "enhanced_fft_mean": enhanced_fft_mean,
            "enhanced_fft_std": enhanced_fft_std,
        })

    save_visualization_and_fft(cropped_image, enhanced_image, chosen_folder, base_name, f"Fidelity {fidelity_level}")

def determine_fidelity_level(increase_ratio):
    if increase_ratio <= 0:
        return 0
    elif increase_ratio <= 1:
        return 1
    elif increase_ratio <= 5:
        return 2
    elif increase_ratio <= 10:
        return 3
    elif increase_ratio <= 15:
        return 4
    else:
        return 5


def process_multiple_images(input_folder, output_folder, csv_path, mu, freq_range=(0.01, 0.49)):
    os.makedirs(output_folder, exist_ok=True)
    results = []
    image_paths = glob.glob(os.path.join(input_folder, "*.png"))

    if not image_paths:
        print(f"No PNG images found in {input_folder}")
        return

    for image_path in image_paths:
        try:
            process_image(image_path, output_folder, mu, freq_range, results)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    input_folder = "../1-0.Train_DB/"
    output_folder = "./250121_output_images_BIMEF"
    csv_path = "./250121_Surface_Integrity_results.csv"
    freq_range = (0.01, 0.49)
    process_multiple_images(input_folder, output_folder, csv_path, mu=0.5, freq_range=freq_range)

    #BIMEF 에서 주로 사용한 sigma 5, mu 0.5를 그대로 사용.
