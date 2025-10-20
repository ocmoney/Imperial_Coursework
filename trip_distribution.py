"""
Trip Distribution using Doubly Constrained Gravity Model with IPF

This script implements the Iterative Proportional Fitting (IPF) algorithm 
to produce a forecasted Origin-Destination (OD) matrix using:
- Trip productions (Table 3)
- Trip attractions (Table 4) 
- Cost matrix (Table 5)
- Deterrence function: f(cij) = cij^(-0.7)
"""

import pandas as pd
import numpy as np

def setup_data():
    """Set up the input data from Tables 3, 4, and 5."""
    
    # Table 3: Trip productions
    productions = {
        'A': 6206.732,
        'B': 7719.092, 
        'C': 10365.199,
        'D': 13237.007,
        'E': 16973.244
    }
    
    # Table 4: Trip attractions
    attractions = {
        'A': 6262.504,
        'B': 7529.096,
        'C': 10275.211, 
        'D': 13126.291,
        'E': 17308.173
    }
    
    # Table 5: Cost matrix
    cost_matrix = np.array([
        [1, 5, 3, 4, 5],  # A
        [5, 1, 5, 1, 3],  # B
        [3, 5, 2, 3, 2],  # C
        [4, 1, 3, 1, 4],  # D
        [5, 3, 2, 4, 2]   # E
    ])
    
    zones = ['A', 'B', 'C', 'D', 'E']
    
    # Convert to arrays for easier computation
    P = np.array([productions[zone] for zone in zones])
    A = np.array([attractions[zone] for zone in zones])
    
    return P, A, cost_matrix, zones

def calculate_deterrence_function(cost_matrix, alpha=0.7):
    """Calculate the deterrence function f(cij) = cij^(-alpha)."""
    return np.power(cost_matrix, -alpha)

def ipf_algorithm(P, A, f_matrix, max_iterations=100, tolerance=0.1):
    """
    Implement the Iterative Proportional Fitting (IPF) algorithm.
    
    Parameters:
    - P: Production vector
    - A: Attraction vector  
    - f_matrix: Deterrence function matrix
    - max_iterations: Maximum number of iterations
    - tolerance: Convergence tolerance (first decimal place = 0.1)
    
    Returns:
    - T: Final OD matrix
    - iterations: Number of iterations to convergence
    """
    
    n_zones = len(P)
    
    # Initialize OD matrix using unconstrained gravity model
    # Tij = Pi * Aj * f(cij) / sum(Ak * f(cik)) for all k
    T = np.zeros((n_zones, n_zones))
    
    for i in range(n_zones):
        denominator = np.sum(A * f_matrix[i, :])
        for j in range(n_zones):
            T[i, j] = P[i] * A[j] * f_matrix[i, j] / denominator
    
    print("Initial unconstrained gravity model:")
    print_matrix(T, "Initial T")
    
    # IPF iterations
    for iteration in range(max_iterations):
        T_old = T.copy()
        
        # Step 1: Balance rows (productions)
        row_sums = np.sum(T, axis=1)
        for i in range(n_zones):
            if row_sums[i] > 0:
                T[i, :] = T[i, :] * P[i] / row_sums[i]
        
        # Step 2: Balance columns (attractions)
        col_sums = np.sum(T, axis=0)
        for j in range(n_zones):
            if col_sums[j] > 0:
                T[:, j] = T[:, j] * A[j] / col_sums[j]
        
        # Check convergence (to first decimal place)
        max_change = np.max(np.abs(T - T_old))
        
        print(f"\nIteration {iteration + 1}:")
        print(f"Maximum change: {max_change:.4f}")
        
        if iteration < 5 or max_change > tolerance:  # Show first few iterations and last
            print_matrix(T, f"T after iteration {iteration + 1}")
            print_balance_check(T, P, A)
        
        if max_change <= tolerance:
            print(f"\nConverged after {iteration + 1} iterations (max change: {max_change:.4f})")
            return T, iteration + 1
    
    print(f"\nDid not converge after {max_iterations} iterations")
    return T, max_iterations

def print_matrix(matrix, title):
    """Print matrix in a formatted way."""
    zones = ['A', 'B', 'C', 'D', 'E']
    print(f"\n{title}:")
    print("     " + "".join(f"{zone:>10}" for zone in zones) + f"{'Total':>10}")
    print("-" * 65)
    
    for i, origin in enumerate(zones):
        row_sum = np.sum(matrix[i, :])
        print(f"{origin:>2}  " + "".join(f"{matrix[i, j]:>10.1f}" for j in range(len(zones))) + f"{row_sum:>10.1f}")
    
    col_sums = np.sum(matrix, axis=0)
    total_sum = np.sum(col_sums)
    print("-" * 65)
    print("Tot " + "".join(f"{col_sums[j]:>10.1f}" for j in range(len(zones))) + f"{total_sum:>10.1f}")

def print_balance_check(T, P, A):
    """Check and print the balance of productions and attractions."""
    row_sums = np.sum(T, axis=1)
    col_sums = np.sum(T, axis=0)
    
    print("\nBalance Check:")
    zones = ['A', 'B', 'C', 'D', 'E']
    print(f"{'Zone':<4} {'Target P':<10} {'Actual P':<10} {'Diff P':<8} {'Target A':<10} {'Actual A':<10} {'Diff A':<8}")
    print("-" * 70)
    
    for i, zone in enumerate(zones):
        p_diff = row_sums[i] - P[i]
        a_diff = col_sums[i] - A[i]
        print(f"{zone:<4} {P[i]:<10.1f} {row_sums[i]:<10.1f} {p_diff:>7.1f} {A[i]:<10.1f} {col_sums[i]:<10.1f} {a_diff:>7.1f}")

def main():
    """Main function to run the trip distribution analysis."""
    
    print("="*80)
    print(" "*25 + "TRIP DISTRIBUTION ANALYSIS")
    print(" "*20 + "Doubly Constrained Gravity Model with IPF")
    print("="*80)
    
    # Setup data
    P, A, cost_matrix, zones = setup_data()
    
    print("\nInput Data:")
    print("="*50)
    
    print("\nTrip Productions (Table 3):")
    for i, zone in enumerate(zones):
        print(f"  Zone {zone}: {P[i]:,.1f}")
    print(f"  Total: {np.sum(P):,.1f}")
    
    print("\nTrip Attractions (Table 4):")
    for i, zone in enumerate(zones):
        print(f"  Zone {zone}: {A[i]:,.1f}")
    print(f"  Total: {np.sum(A):,.1f}")
    
    print("\nCost Matrix (Table 5):")
    print("     " + "".join(f"{zone:>6}" for zone in zones))
    for i, origin in enumerate(zones):
        print(f"{origin:>2}  " + "".join(f"{cost_matrix[i, j]:>6.0f}" for j in range(len(zones))))
    
    # Calculate deterrence function
    f_matrix = calculate_deterrence_function(cost_matrix)
    
    print(f"\nDeterrence Function f(cij) = cij^(-0.7):")
    print("     " + "".join(f"{zone:>8}" for zone in zones))
    for i, origin in enumerate(zones):
        print(f"{origin:>2}  " + "".join(f"{f_matrix[i, j]:>8.4f}" for j in range(len(zones))))
    
    # Run IPF algorithm
    print(f"\n" + "="*80)
    print(" "*25 + "IPF ALGORITHM ITERATIONS")
    print("="*80)
    
    final_T, iterations = ipf_algorithm(P, A, f_matrix)
    
    # Final results
    print(f"\n" + "="*80)
    print(" "*30 + "FINAL RESULTS")
    print("="*80)
    
    print_matrix(final_T, "Final OD Matrix")
    print_balance_check(final_T, P, A)
    
    # Save results
    df_od = pd.DataFrame(final_T, index=zones, columns=zones)
    df_od.to_csv('od_matrix_final.csv')
    
    print(f"\nResults saved to 'od_matrix_final.csv'")
    print(f"Algorithm converged in {iterations} iterations")
    
    return final_T

if __name__ == "__main__":
    final_od_matrix = main()
