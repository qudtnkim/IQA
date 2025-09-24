import pandas as pd
import numpy as np
import argparse
from typing import Dict

def calculate_structural_variability(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    """
    Calculates Structural Variability (S) based on the provided text.
    'S' is defined as the Euclidean distance from the origin in the 2D plane
    formed by the homogeneity index and the product index.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col_map (Dict[str, str]): Dictionary mapping standard names to actual column names.
                                  e.g., {'homogeneity': 'H_Z', 'product': 'HS'}

    Returns:
        pd.DataFrame: DataFrame with a new 'Structural_Variability_S' column.
    """
    homogeneity_col = col_map['homogeneity']
    product_col = col_map['product']
    
    print(f"Calculating 'S' using columns: '{homogeneity_col}' and '{product_col}'")
    
    # S = sqrt(x^2 + y^2), where x is the product index and y is the homogeneity index.
    df['Structural_Variability_S'] = np.sqrt(df[product_col]**2 + df[homogeneity_col]**2)
    
    return df

def calculate_final_iqa_score(df: pd.DataFrame, col_map: Dict[str, str], epsilon: float = 1e-6) -> pd.DataFrame:
    """
    Calculates the final IQA score using the formula: IQA = Clarity + 1/S.

    Args:
        df (pd.DataFrame): DataFrame containing 'Clarity' and 'Structural_Variability_S'.
        col_map (Dict[str, str]): Dictionary mapping standard names to actual column names.
                                  e.g., {'clarity': 'Clarity'}
        epsilon (float): A small constant to prevent division by zero.

    Returns:
        pd.DataFrame: DataFrame with 'Normalized_Clarity' and 'Final_IQA_Score' columns.
    """
    clarity_col = col_map['clarity']
    
    # 1. Normalize the Clarity score to a [0, 1] range to act as the base score.
    min_clarity = df[clarity_col].min()
    max_clarity = df[clarity_col].max()
    df['Normalized_Clarity'] = (df[clarity_col] - min_clarity) / (max_clarity - min_clarity)
    
    # 2. Apply the final formula: IQA_Score = Clarity + 1/S
    # The 1/S term acts as a powerful non-linear penalty/bonus.
    df['Final_IQA_Score'] = df['Normalized_Clarity'] + (1 / (df['Structural_Variability_S'] + epsilon))
    
    print("Final IQA score calculated successfully.")
    return df

def main(args):
    """
    Main function to load data, calculate scores, and save results.
    """
    try:
        df = pd.read_csv(args.input_csv)
        print(f"Successfully loaded {args.input_csv} with {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: The file '{args.input_csv}' was not found.")
        return

    # --- Define Column Mappings ---
    # This dictionary maps the conceptual names from your text to the actual column names in the CSV.
    # Please adjust these if your CSV column names are different.
    column_mapping = {
        'homogeneity': 'H_Z',
        'product': 'HS',
        'clarity': 'Clarity'
    }

    # --- Run the IQA Pipeline ---
    # 1. Calculate Structural Variability (S)
    df = calculate_structural_variability(df, column_mapping)
    
    # 2. Calculate the final composite IQA score
    df = calculate_final_iqa_score(df, column_mapping)
    
    # --- Save the Results ---
    df.to_csv(args.output_csv, index=False)
    print(f"Results successfully saved to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate a composite IQA score from clarity, homogeneity, and surface integrity data.")
    
    parser.add_argument("-i", "--input_csv", required=True, help="Path to the input CSV file containing H_Z, HS, and Clarity columns.")
    parser.add_argument("-o", "--output_csv", required=True, help="Path to save the output CSV file with the final scores.")
    
    args = parser.parse_args()
    main(args)
