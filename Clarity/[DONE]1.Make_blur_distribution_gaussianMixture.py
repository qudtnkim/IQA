import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

def analyze_s_frequency_gmm(csv_path, output_csv_path, n_components=2):
    """
    Analyze S Frequency Ratios using GMM and calculate Strong Threshold.
    
    Args:
        csv_path: Input CSV file containing at least ["Image Name", "S Frequency Ratio"].
        output_csv_path: Where to save the result CSV with an added "Cluster" column.
        n_components: Number of Gaussian components (default: 2 for bimodal).

    Returns:
        Strong Threshold (float): The upper bound of the 3σ range for the smaller-mean component.
    """

    # 1) Load CSV
    df = pd.read_csv(csv_path)

    # 2) Extract data for GMM (reshape to (n,1))
    X = df["S Frequency Ratio"].values.reshape(-1, 1)

    # 3) Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)

    # Means, variances, and weights
    means = gmm.means_.flatten()
    covars = gmm.covariances_.flatten()  # for 1D, covariance is just a 1D array
    weights = gmm.weights_.flatten()

    # 4) Assign each sample to a cluster
    cluster_labels = gmm.predict(X)  # array of 0 or 1, etc.
    df["Cluster"] = cluster_labels

    # 5) Print summary
    print("=== GMM Results ===")
    for i in range(n_components):
        print(f"Component {i}: mean={means[i]:.4f}, var={covars[i]:.4f}, weight={weights[i]:.4f}")

    # (추가 기능) 평균이 더 작은 성분의 3σ, 6σ 범위 계산
    idx_small = np.argmin(means)             # 평균이 더 작은 index
    mu_small = means[idx_small]
    var_small = covars[idx_small]
    sigma_small = np.sqrt(var_small)
 
    range_3sigma = (mu_small - 3*sigma_small, mu_small + 3*sigma_small)

    print("\n[INFO] Smaller-mean component index:", idx_small)
    print(f"[INFO] => mu={mu_small:.4f}, sigma={sigma_small:.4f}")
    print(f"[INFO] => 3σ range = {range_3sigma[0]:.4f} ~ {range_3sigma[1]:.4f}")

    # Strong Threshold: 3σ 범위의 상한값
    strong_threshold = range_3sigma[1]
    print(f"[INFO] Strong Threshold (3σ upper bound): {strong_threshold:.4f}")

    # 6) Save the updated DataFrame to a new CSV
    df.to_csv(output_csv_path, index=False)
    print(f"[INFO] Updated CSV with 'Cluster' column saved to {output_csv_path}")

    # 7) (Optional) Plot histogram & GMM PDF
    plt.figure(figsize=(8, 5))
    plt.hist(X, bins=30, density=True, alpha=0.5, color="gray", edgecolor="black", label="Histogram")

    # Plot the mixture PDF across a range of x-values
    x_min, x_max = X.min(), X.max()
    x_plot = np.linspace(x_min, x_max, 300).reshape(-1, 1)
    logprob = gmm.score_samples(x_plot)
    pdf = np.exp(logprob)
    plt.plot(x_plot, pdf, 'r-', label="GMM Mixture PDF")

    # Plot each component individually
    responsibilities = gmm.predict_proba(x_plot)  # shape: (300, n_components)
    pdf_individual = responsibilities * pdf[:, np.newaxis]

    for i in range(n_components):
        plt.plot(
            x_plot,
            pdf_individual[:, i],
            linestyle="--",
            linewidth=1.5,
            label=f"GMM Component {i}"
        )
    
    # 수직선으로 3σ, 6σ 범위를 표시 (작은 평균 성분 기준)
    ymin, ymax = plt.ylim()
    plt.vlines([range_3sigma[0], range_3sigma[1]], ymin, ymax * 0.3, 
               colors="blue", linestyles=":", label="3σ range")

    plt.xlabel("High-Frequency to Low-Frequency Ratio")
    plt.ylabel("Density")
    plt.title("HLFR Histogram + GMM Fit (with 3σ Range)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return strong_threshold


if __name__ == "__main__":
    # Example usage:
    input_csv = "250120_S_CHANNEL_Clarity_result.csv"     # CSV with ["Image Name", "S Frequency Ratio"]
    output_csv = "whole_train_gmm.csv"

    strong_threshold = analyze_s_frequency_gmm(
        csv_path=input_csv,
        output_csv_path=output_csv,
        n_components=2
    )

    print(f"\nFinal Strong Threshold: {strong_threshold:.4f}")
