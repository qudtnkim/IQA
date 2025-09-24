# 이미지 선명도 기반 데이터 정제 파이프라인 (Image Clarity-Based Data Filtering Pipeline)
이 리포지토리는 머신러닝 모델 학습에 사용될 이미지 데이터셋에서 흐릿한(blurry) 이미지를 자동으로 식별하고 필터링하기 위한 파이프라인을 제공합니다. 고속 푸리에 변환(FFT)을 사용하여 각 이미지의 선명도를 정량화하고, 가우시안 혼합 모델(GMM)을 통해 데이터 분포에 기반한 최적의 임계값(threshold)을 자동으로 결정합니다.

이 파이프라인을 통해 데이터셋의 품질을 향상시켜 모델의 학습 성능과 안정성을 높일 수 있습니다.

## 주요 기능 (Key Features)
FFT 기반 선명도 측정: 각 이미지의 주파수 성분을 분석하여 고주파와 저주파 에너지 비율(HLFR)을 계산하고, 이를 선명도 점수로 활용합니다.

GMM을 이용한 자동 임계값 결정: 전체 데이터의 선명도 점수 분포를 분석하여 '선명한 그룹'과 '흐릿한 그룹'을 분리하는 통계적으로 유의미한 임계값을 자동으로 찾아냅니다.

자동 데이터 필터링: 결정된 임계값을 기준으로 원본 데이터셋에서 흐릿한 이미지를 자동으로 분류하고 별도의 폴더로 분리합니다.

## 파이프라인 워크플로우 (Pipeline Workflow)
이 리포지토리는 3단계의 워크플로우로 구성되어 있습니다.



### 1단계: 선명도 점수 계산
스크립트: Norm_Clarity_IQA.py

역할: 입력된 이미지 폴더의 모든 이미지에 대해 선명도 점수(HLFR)를 계산하고, 결과를 CSV 파일로 저장합니다. 이 파일에는 각 이미지의 이름과 원본 점수(raw score), 정규화된 점수(normalized score)가 포함됩니다.

### 2단계: 최적 임계값 결정
스크립트: Make_blur_distribution_gaussianMixture.py

역할: 1단계에서 생성된 CSV 파일을 입력받아 선명도 점수 분포를 GMM으로 분석합니다. 이를 통해 '선명한 이미지'와 '흐릿한 이미지' 두 그룹을 가장 잘 나누는 최적의 선명도 임계값을 계산하여 출력합니다.

### 3단계: 이미지 자동 필터링
스크립트: NORM_CLARITY_IMAGE_RETURN.py

역할: 2단계에서 찾은 임계값을 사용하여, 선명도 점수가 이 기준에 미치지 못하는 이미지(흐릿한 이미지)를 지정된 폴더로 복사하거나 이동시킵니다. 이를 통해 최종적으로 정제된 '선명한 이미지' 데이터셋을 얻을 수 있습니다.



## 사용방법
# 1. 데이터셋의 모든 이미지에 대해 선명도 점수를 계산하고 CSV 파일로 저장합니다.
python Norm_Clarity_IQA.py --input_dir "path/to/your/images" --output_csv "clarity_scores.csv"

# 2. 생성된 CSV 파일을 바탕으로 GMM을 실행하여 최적의 임계값을 찾습니다.
python Make_blur_distribution_gaussianMixture.py --input_csv "clarity_scores.csv"

# 3. 위에서 찾은 임계값을 사용하여 흐릿한 이미지를 별도 폴더로 분리합니다.
python NORM_CLARITY_IMAGE_RETURN.py --image_dir "path/to/your/images" --scores_csv "clarity_scores.csv" --threshold <calculated_threshold>
