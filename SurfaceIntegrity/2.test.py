import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.integrate import trapz
import glob
import os

def bimef(image, mu=0.5, a=-0.3293, b=1.1258, lambda_=0.5, sigma=5): #sigma=5
    """
    BIMEF 알고리즘 적용
    #mu =    #BIMEF에서 weight map의 gamma coeff임
    #각 픽셀의 조명 정보 t_b_smooth가 결과이미지에 얼마나 강하게 반영될지 결정
    #t_b smoothing 과정에 쓰는 Gaussian Blur filter의 sigma를 크게하면 넓은 영역의 평균을 계산함
    # 큰 시그마는 넓은 영역에 영향을 줌.
    """
    t_b = np.max(image, axis=2)  # Scene illumination map
    t_b_smooth = cv2.GaussianBlur(t_b, (sigma * 2 + 1, sigma * 2 + 1), sigma) 
    weight_map = np.clip(t_b_smooth ** mu, 0, 1)
    enhanced_image = image * weight_map[..., np.newaxis]
    
    # Normalize the output to [0, 255]
    enhanced_image = (enhanced_image - enhanced_image.min()) / (enhanced_image.max() - enhanced_image.min()) * 255
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
    
    return enhanced_image

def compute_psd(image, fs=1.0, percent_nperseg=0.2, overlap_ratio=0.5, scaling='density'):
    """
    Welch 방법으로 Power Spectral Density (PSD) 계산
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flattened_signal = gray.flatten().astype(np.float64)
    flattened_signal -= np.mean(flattened_signal)

    data_length = flattened_signal.size
    nperseg = int(data_length * percent_nperseg)
    nperseg = max(1, nperseg)
    noverlap = int(nperseg * overlap_ratio)

    freqs, power = welch(flattened_signal, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling=scaling)
    return freqs, power

def extract_triangle(image, ratio_w=90/1255, ratio_h=100/1080):
    """
    비율에 따라 양옆/위아래를 잘라내어 삼각형 영역 제거한 사각형으로 크롭
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

    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image

def process_image(image_path, output_folder, hq_folder, lq_folder, mu=0.5, freq_range=(0.1, 0.49)): 
    """
    이미지 품질 평가 및 BIMEF 적용 후:
    - Cropped vs Enhanced PSD 비교
    - 증가 폭 기준으로 HQ / LQ 분류

    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # (1) 삼각형 영역 제거
    cropped_image = extract_triangle(image)

    # (2) BIMEF 적용
    enhanced_image = bimef(cropped_image, mu=mu)

    # (3) PSD 계산
    freqs_cropped, power_cropped = compute_psd(cropped_image)
    freqs_enhanced, power_enhanced = compute_psd(enhanced_image)

    # (4) 주파수 범위 제한 & 적분
    f_min, f_max = freq_range
    mask_cropped = (freqs_cropped >= f_min) & (freqs_cropped <= f_max)
    area_cropped = trapz(power_cropped[mask_cropped], freqs_cropped[mask_cropped])

    mask_enhanced = (freqs_enhanced >= f_min) & (freqs_enhanced <= f_max)
    area_enhanced = trapz(power_enhanced[mask_enhanced], freqs_enhanced[mask_enhanced])

    increase_ratio = (area_enhanced - area_cropped) / area_cropped
    print(f"[INFO] freq range {f_min}~{f_max} => Increase Ratio: {increase_ratio:.4f}")

    # (5) HQ / LQ 판단
    base_name = os.path.basename(image_path).split('.')[0]
    if increase_ratio > 0.2:  # 증가 폭 기준 (0.2는 예시값, 필요시 조정)
        chosen_folder = lq_folder
        label_str = "LQ"
        print("Upper value:",area_enhanced - area_cropped)
        print("Lower value:",area_cropped)
    else:
        chosen_folder = hq_folder
        label_str = "HQ"
        print("Upper value:",area_enhanced - area_cropped)
        print("Lower value:",area_cropped)

    # (6) 결과물 저장
    # 6-1) Enhanced image
    enhanced_image_path = os.path.join(chosen_folder, f"{base_name}.png")
    cv2.imwrite(enhanced_image_path, enhanced_image)
    #print(f"[{label_str}] Enhanced image saved: {enhanced_image_path}")

    # 6-2) PSD 비교 그래프
    psd_output_path = os.path.join(output_folder, f"{base_name}_psd_plot.png")
    plt.figure(figsize=(8, 6))
    plt.title('PSD Comparison')
    plt.semilogy(freqs_cropped, power_cropped, label='Cropped PSD', linestyle='--')
    plt.semilogy(freqs_enhanced, power_enhanced, label='Enhanced PSD', linestyle='-')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.legend()
    plt.grid()
    plt.savefig(psd_output_path)
    plt.close()
    #print(f"PSD Comparison saved: {psd_output_path}")

    # 6-3) Visualization of Original Cropped vs Enhanced
    visualization_path = os.path.join(output_folder, f"{base_name}_visualization.png")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Cropped Image')
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Enhanced Image (BIMEF)')
    plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(visualization_path)
    plt.close()
    #print(f"Visualization saved: {visualization_path}")

def process_multiple_images(input_folder, output_folder, mu=0.5, freq_range=(0.1, 0.49)):
    os.makedirs(output_folder, exist_ok=True)
    hq_folder = os.path.join(output_folder, "HQ")
    lq_folder = os.path.join(output_folder, "LQ")
    os.makedirs(hq_folder, exist_ok=True)
    os.makedirs(lq_folder, exist_ok=True)

    image_paths = glob.glob(os.path.join(input_folder, "*.png"))
    if not image_paths:
        print(f"No PNG images found in {input_folder}")
        return

    for image_path in image_paths:
        try:
            process_image(
                image_path,
                output_folder=output_folder,
                hq_folder=hq_folder,
                lq_folder=lq_folder,
                mu=mu,
                freq_range=freq_range
            )
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

if __name__ == "__main__":
    input_folder = "../1.Exclude_Images_workPlace/"
    output_folder = "./out"
    freq_range = (0.01, 0.49)

    process_multiple_images(input_folder, output_folder, mu=0.5, freq_range=freq_range)
    #BIMEF 에서 주로 사용한 sigma 5, mu 0.5를 그대로 사용.
