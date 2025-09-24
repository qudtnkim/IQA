## 최종 IQA 점수 산출 스크립트 (Final IQA Score Calculation)
이 스크립트는 IQA 파이프라인의 최종 단계로, 개별적으로 측정된 이미지 품질 지표들인 **선명도(Clarity), 균일성(Homogeneity), 표면 무결성(Surface Integrity)**를 하나의 
의미 있는 **종합 점수(Composite IQA Score)**로 통합합니다.

## 핵심 원리 (Core Principle)
이 모델의 설계는 다음과 같은 핵심적인 임상 원칙을 따릅니다:

"선명도는 이미지 품질의 기본선(baseline)을 제공하지만, 구조적 결함은 결정적인 결함으로 작용한다."

따라서 이 모델은 모든 품질 차원을 동일하게 취급하지 않고, 

구조적 결함에 대해 강력한 비선형적 페널티를 부과하도록 의도적으로 설계되었습니다. 

이는 단 하나의 결정적 결함이 선명한 이미지를 무용지물로 만드는 인간의 인지 과정을 모방합니다.


## 방법론 (Methodology)
최종 점수는 다음의 2단계 과정을 통해 계산됩니다.

### 1. 구조 변동성 (Structural Variability, S) 계산
Homogeneity와 Surface Integrity 지표는 **구조 변동성(S)**이라는 단일 통합 지표로 결합됩니다.

S는 구조적 결함의 총량을 나타내며, Homogeneity Index (H_Z)를 y축, Homogeneity * Surface Integrity (HS)를 x축으로 하는 2D 평면에서 원점(0,0)으로부터 의 유클리드 거리로 계산됩니다.

S 값이 클수록 이미지에 구조적 복잡성과 결함이 많다는 것을 의미합니다.

​
 
### 2. 최종 IQA 공식 적용
최종 IQA 점수는 Clarity를 기본 점수로, S를 페널티 항으로 사용하여 다음 공식에 따라 계산됩니다.

Normalized_Clarity: 0과 1 사이로 정규화된 선명도 점수로, 안정적인 기본 점수를 제공합니다.

1 / (S + ε): 비선형적 페널티 및 보너스 항입니다.

S가 작을 경우 (결함이 적음): 1/S 항은 매우 큰 보너스 점수가 되어 구조적으로 뛰어난 이미지에 높은 점수를 부여합니다.

S가 클 경우 (결함이 많음): 1/S 항은 0에 빠르게 수렴하여, Clarity 점수를 효과적으로 무효화시키는 강력한 페널티로 작용합니다.

## 사용 방법 (Usage)
### 입력 파일 요구사항
스크립트를 실행하기 전에, 입력 CSV 파일에 반드시 다음 열(column)들이 포함되어 있어야 합니다. 열 이름은 정확히 일치해야 합니다.

H_Z : Homogeneity Index

HS : Homogeneity Index * Surface Integrity Index

Clarity : 선명도 점수
