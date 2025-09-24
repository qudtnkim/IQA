import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def analyze_clarity_with_gmm(
    input_csv_path: str,
    output_csv_path: str,
    column_name: str = "S Frequency Ratio",
    n_components: int = 2
) -> float:
    """
    Analyzes image clarity scores using a Gaussian Mixture Model (GMM) to
    identify a threshold for separating clear vs. unclear images.

    This function assumes the distribution of clarity scores is bimodal,
    representing two primary groups (e.g., 'blurry' and 'clear' images).
    It fits a GMM with two components and calculates a threshold based on the
    statistical properties of the component with the lower mean score.

    Args:
        input_csv_path (str): Path to the input CSV file. The file must contain
                              a column with clarity scores.
        output_csv_path (str): Path to save the output CSV. The saved file will
                               include a new 'Cluster' column.
        column_name (str): The name of the column containing the clarity scores
                           (e.g., "S Frequency Ratio").
        n_components (int): The number of Gaussian components to fit. Defaults to 2.

    Returns:
        float: The calculated clarity threshold, defined as the upper bound of the
               3-sigma range of the Gaussian component with the smaller mean.
    """

    # --- 1. Load and Prepare Data ---
    print(f"Loading data from '{input_csv_path}'...")
    try:
        df = pd.read_csv(input_csv_path)
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the CSV file.")
    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' was not found.")
        return -1.0
    
    # Extract the target column and reshape for sklearn
    X = df[column_name].values.reshape(-1, 1)

    # --- 2. Fit Gaussian Mixture Model ---
    print("Fitting Gaussian Mixture Model...")
    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    gmm.fit(X)

    # Assign cluster labels to the original data
    cluster_labels = gmm.predict(X)
    df["Cluster"] = cluster_labels

    # --- 3. Extract GMM Parameters and Print Summary ---
    means = gmm.means_.flatten()
    std_devs = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    print("\n" + "="*25)
    print(" GMM Fit Results")
    print("="*25)
    for i in range(n_components):
        print(f"  Component {i}:")
        print(f"    - Mean      = {means[i]:.4f}")
        print(f"    - Std. Dev. = {std_devs[i]:.4f}")
        print(f"    - Weight    = {weights[i]:.4f}")

    # --- 4. Calculate Clarity Threshold ---
    # Identify the component with the smaller mean. This component is assumed to
    # represent the cluster of 'blurry' or 'low-clarity' images.
    blurry_cluster_index = np.argmin(means)
    mu_blurry = means[blurry_cluster_index]
    sigma_blurry = std_devs[blurry_cluster_index]

    # The threshold is defined as the upper bound of the 3-sigma range for this component.
    # According to the 3-sigma rule, this covers ~99.7% of the data in this cluster.
    clarity_threshold = mu_blurry + 3 * sigma_blurry

    print("\n" + "="*30)
    print(" Clarity Threshold Calculation")
    print("="*30)
    print(f"  Identifying component with the smaller mean (assumed 'blurry')...")
    print(f"  Blurry Component Index: {blurry_cluster_index}")
    print(f"  Mean (μ): {mu_blurry:.4f}, Std. Dev. (σ): {sigma_blurry:.4f}")
    print(f"  3-Sigma Range: ({mu_blurry - 3 * sigma_blurry:.4f}, {mu_blurry + 3 * sigma_blurry:.4f})")
    print(f"  => Calculated Threshold (μ + 3σ): {clarity_threshold:.4f}")
    
    # --- 5. Save Results ---
    df.to_csv(output_csv_path, index=False)
    print(f"\n[SUCCESS] Data with cluster assignments saved to '{output_csv_path}'.")

    # --- 6. Visualize the Results (Publication Quality Plot) ---
    plt.style.use('seaborn-v0_8-paper')
    plt.figure(figsize=(10, 6))

    # Plot histogram of the data
    plt.hist(X, bins=50, density=True, alpha=0.6, color="lightgray", label="Data Histogram")

    # Plot the individual Gaussian components
    x_plot = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
    for i in range(n_components):
        # Calculate the PDF for each component, scaled by its weight
        pdf = weights[i] * norm(means[i], std_devs[i]).pdf(x_plot)
        linestyle = '--' if i == blurry_cluster_index else '-.'
        plt.plot(x_plot, pdf, linestyle=linestyle, label=f"GMM Component {i}")

    # Plot the total GMM probability density function
    logprob = gmm.score_samples(x_plot)
    plt.plot(x_plot, np.exp(logprob), 'r-', linewidth=2, label="Total GMM PDF")
    
    # Highlight the calculated threshold with a vertical line
    plt.axvline(
        clarity_threshold,
        color='black',
        linestyle=':',
        linewidth=2,
        label=f"Clarity Threshold = {clarity_threshold:.2f}"
    )

    plt.title(f"Distribution of '{column_name}' and GMM Fit", fontsize=16)
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return clarity_threshold


# --- Main execution block ---
if __name__ == "__main__":
    # --- Configuration ---
    # Define the path to your input CSV file
    INPUT_CSV = "250120_S_CHANNEL_Clarity_result.csv"
    
    # Define the path for the output CSV file
    OUTPUT_CSV = "whole_train_clarity_gmm_results.csv"
    
    # Specify the name of the column containing the clarity metric
    CLARITY_COLUMN = "S Frequency Ratio"

    # --- Run Analysis ---
    final_threshold = analyze_clarity_with_gmm(
        input_csv_path=INPUT_CSV,
        output_csv_path=OUTPUT_CSV,
        column_name=CLARITY_COLUMN,
        n_components=2
    )
    
    if final_threshold != -1.0:
        print(f"\nAnalysis complete. The final calculated threshold is: {final_threshold:.4f}")
