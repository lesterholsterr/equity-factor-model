"""
This module provides functionality for performing Singular Value Decomposition (SVD) 
on a set of factors to extract orthogonal factors and apply the learned transformation 
to new data. It includes functions for extracting SVD factors from training data and 
applying the learned SVD transformation to test data.

Functions:
- svd_factor_extraction: Perform SVD on training data to extract orthogonal factors, 
    singular values, signal weights, and scaling information.
- apply_svd_factors: Apply the learned SVD transformation to test data using the 
    signal weights and scaling information from training.

Dependencies:
- pandas
- numpy
- scipy (optional, used as a fallback for SVD computation)
"""

import pandas as pd
import numpy as np

def svd_factor_extraction(data: pd.DataFrame, top_factors: list, n_factors: int = 15) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """
        Perform Singular Value Decomposition (SVD) on the specified factors to extract 
        new orthogonal factors and compute signal weights.

        Parameters:
        - data (pd.DataFrame): DataFrame containing the original factor data.
        - top_factors (list): List of column names representing the factors to use for SVD.
        - n_factors (int, optional): Number of SVD factors to extract. Default is 15.

        Returns:
        - svd_factors_df (pd.DataFrame): DataFrame containing the new factors extracted from SVD.
        - S (np.ndarray): Array of singular values from the decomposition.
        - signal_weights (pd.DataFrame): DataFrame containing the weights of each original 
            signal for each SVD factor.
        - scaling_info (pd.DataFrame): DataFrame containing the mean and standard deviation 
            of each factor, used for standardization.

        Raises:
        - np.linalg.LinAlgError: If the SVD computation fails, a fallback to scipy's SVD is used.

        Example:
        >>> if __name__ == "__main__":
        >>>     # Assuming `data` is a DataFrame and `top_factors` is a dictionary with a 'factor' key
        >>>     svd_factors, S, signal_weights, scaling_info = svd_factor_extraction(
        >>>         data[data['yyyymm'] < 201501], 
        >>>         top_factors['factor'], 
        >>>         n_factors=12
        >>>     )
        """
        factor_data = data[top_factors].copy()
        
        # Store means and standard deviations for later use with test data
        means = factor_data.mean()
        stds = factor_data.std()
        
        # Fill NAs and standardize
        factor_data = factor_data.fillna(means)
        factor_data_standardized = (factor_data - means) / stds
        
        try:
                U, S, Vt = np.linalg.svd(factor_data_standardized.values, full_matrices=False)
        except np.linalg.LinAlgError as e:
                print(f"SVD calculation error: {e}")
                from scipy import linalg
                U, S, Vt = linalg.svd(factor_data_standardized.values, full_matrices=False)
        
        # Create SVD factors
        selected_factors = U[:, :n_factors] @ np.diag(S[:n_factors])
        svd_factors_df = pd.DataFrame(selected_factors, columns=[f'SVD_Factor_{i+1}' for i in range(n_factors)])
        svd_factors_df.index = factor_data.index
        
        # Extract signal weights (V matrix from SVD, transpose of Vt)
        signal_weights = pd.DataFrame(
                Vt[:n_factors, :].T,  # Transpose to get factors in columns
                columns=[f'SVD_Factor_{i+1}' for i in range(n_factors)],
                index=top_factors
        )
        
        # Also save the scaling information to apply the same transformation to test data
        scaling_info = pd.DataFrame({
                'mean': means,
                'std': stds
        })
        
        return svd_factors_df, S, signal_weights, scaling_info

def apply_svd_factors(test_data: pd.DataFrame, top_factors: list, signal_weights: pd.DataFrame, 
                                            scaling_info: pd.DataFrame, n_factors: int = 15) -> pd.DataFrame:
        """
        Apply the learned SVD transformation to new test data.

        Parameters:
        - test_data (pd.DataFrame): DataFrame containing the test data.
        - top_factors (list): List of column names representing the factors used in training.
        - signal_weights (pd.DataFrame): DataFrame of signal weights from svd_factor_extraction.
        - scaling_info (pd.DataFrame): DataFrame with mean and standard deviation for standardization.
        - n_factors (int, optional): Number of SVD factors to create. Default is 15.

        Returns:
        - test_svd_factors (pd.DataFrame): DataFrame containing the SVD factors for test data.

        Example:
        >>>     # Generate test SVD factors for data after 2015
        >>>     test_data = data[data['yyyymm'] >= 201501].copy()
        >>>     test_svd_factors = apply_svd_factors(
        >>>         test_data, 
        >>>         top_factors['factor'], 
        >>>         signal_weights, 
        >>>         scaling_info, 
        >>>         n_factors=12
        >>>     )
        """
        # Extract test data for the top factors
        test_factor_data = test_data[top_factors].copy()
        
        # Use the same means and stds from training data
        means = scaling_info['mean']
        stds = scaling_info['std']
        
        # Handle missing values and standardize using training data statistics
        test_factor_data = test_factor_data.fillna(means)
        test_standardized = (test_factor_data - means) / stds
        
        # Apply the weights to get the test SVD factors
        test_svd_values = test_standardized.values @ signal_weights.values
        
        # Create DataFrame with the new factors
        test_svd_factors = pd.DataFrame(
                test_svd_values,
                columns=[f'SVD_Factor_{i+1}' for i in range(n_factors)],
                index=test_data.index
        )
        
        return test_svd_factors
