"""
Mode Choice Model Implementation

This script applies a multinomial logit mode choice model to the OD matrix
from Part 2 to calculate mode-specific OD matrices for walk, rail, and car.

Utility functions:
VWALK = -0.6 * tWALK
VRAIL = 0.1 - 0.5 * tRAIL - 2 * cRAIL - 0.7 * tWAIT
VCAR  = 0.2 - 0.4 * tCAR - 1.5 * cCAR - 3 * cPARK
"""

import pandas as pd
import numpy as np
import math

def load_od_matrix():
    """Load the OD matrix from Part 2."""
    try:
        od_matrix = pd.read_csv('od_matrix_final.csv', index_col=0)
        return od_matrix.values
    except FileNotFoundError:
        print("Warning: od_matrix_final.csv not found. Using example OD matrix.")
        # Use the final OD matrix from Part 2
        return np.array([
            [1854.7, 545.0, 1218.3, 1099.0, 1489.8],
            [568.3, 1589.7, 805.5, 2741.8, 2013.8],
            [1238.1, 785.0, 2330.7, 1936.1, 4075.3],
            [1128.4, 2699.8, 1956.0, 4656.5, 2796.4],
            [1473.0, 1909.6, 3964.9, 2692.9, 6932.9]
        ])

def setup_mode_data():
    """Set up all the mode-specific data from Tables 6-12."""
    
    zones = ['A', 'B', 'C', 'D', 'E']
    
    # Table 6: Walking travel duration (minutes)
    t_walk = np.array([
        [0, 35, 30, 35, 40],
        [35, 0, 35, 75, 40],
        [30, 35, 0, 40, 75],
        [40, 75, 40, 0, 50],
        [35, 40, 75, 50, 0]
    ])
    
    # Table 7: Rail travel duration (minutes)
    t_rail = np.array([
        [0, 15, 15, 20, 25],
        [15, 0, 20, 35, 20],
        [15, 20, 0, 25, 30],
        [20, 35, 25, 0, 40],
        [25, 20, 30, 40, 0]
    ])
    
    # Table 8: Car travel duration (minutes)
    t_car = np.array([
        [0, 18, 15, 15, 20],
        [18, 0, 17, 35, 18],
        [15, 17, 0, 22, 30],
        [15, 35, 22, 0, 20],
        [20, 18, 30, 20, 0]
    ])
    
    # Table 9: Rail ticket cost (£)
    c_rail = np.array([
        [0, 1, 1, 1, 1],
        [1, 0, 1.5, 2.5, 1.5],
        [1, 1.5, 0, 1.5, 2],
        [1, 2.5, 1.5, 0, 2.5],
        [1, 1.5, 2, 2.5, 0]
    ])
    
    # Table 10: Car monetary cost (£)
    c_car = np.array([
        [0, 2, 2, 2, 2],
        [2, 0, 3, 4, 3],
        [2, 3, 0, 3, 4],
        [2, 4, 3, 0, 3],
        [2, 3, 4, 3, 0]
    ])
    
    # Table 11: Rail waiting time (minutes)
    t_wait = np.array([
        [0, 10, 10, 10, 10],
        [10, 0, 20, 10, 20],
        [10, 20, 0, 20, 10],
        [10, 10, 20, 0, 10],
        [10, 20, 10, 10, 0]
    ])
    
    # Table 12: Car parking cost (£)
    c_park = np.array([
        [0, 2.5, 2.5, 2.5, 2.5],
        [2.5, 0, 2.5, 2.5, 2.5],
        [2.5, 2.5, 0, 2.5, 2.5],
        [2.5, 2.5, 2.5, 0, 2.5],
        [2.5, 2.5, 2.5, 2.5, 0]
    ])
    
    return {
        'zones': zones,
        't_walk': t_walk,
        't_rail': t_rail,
        't_car': t_car,
        'c_rail': c_rail,
        'c_car': c_car,
        't_wait': t_wait,
        'c_park': c_park
    }

def calculate_utilities(data):
    """Calculate utility matrices for each mode."""
    
    # VWALK = -0.6 * tWALK
    V_walk = -0.6 * data['t_walk']
    
    # VRAIL = 0.1 - 0.5 * tRAIL - 2 * cRAIL - 0.7 * tWAIT
    V_rail = 0.1 - 0.5 * data['t_rail'] - 2 * data['c_rail'] - 0.7 * data['t_wait']
    
    # VCAR = 0.2 - 0.4 * tCAR - 1.5 * cCAR - 3 * cPARK
    V_car = 0.2 - 0.4 * data['t_car'] - 1.5 * data['c_car'] - 3 * data['c_park']
    
    return V_walk, V_rail, V_car

def calculate_mode_probabilities(V_walk, V_rail, V_car):
    """Calculate mode choice probabilities using multinomial logit model."""
    
    n_zones = V_walk.shape[0]
    P_walk = np.zeros((n_zones, n_zones))
    P_rail = np.zeros((n_zones, n_zones))
    P_car = np.zeros((n_zones, n_zones))
    
    for i in range(n_zones):
        for j in range(n_zones):
            # if i == j:  # Intrazonal trips
            #     # For intrazonal trips, only walking is available
            #     P_walk[i, j] = 1.0
            #     P_rail[i, j] = 0.0
            #     P_car[i, j] = 0.0
            # else:
                # Calculate exponentials
                exp_walk = math.exp(V_walk[i, j])
                exp_rail = math.exp(V_rail[i, j])
                exp_car = math.exp(V_car[i, j])
                
                # Sum of exponentials
                sum_exp = exp_walk + exp_rail + exp_car
                
                # Calculate probabilities
                P_walk[i, j] = exp_walk / sum_exp
                P_rail[i, j] = exp_rail / sum_exp
                P_car[i, j] = exp_car / sum_exp
    
    return P_walk, P_rail, P_car

def print_matrix(matrix, title, zones):
    """Print matrix in a formatted way."""
    print(f"\n{title}:")
    print("     " + "".join(f"{zone:>10}" for zone in zones) + f"{'Total':>10}")
    print("-" * (10 * len(zones) + 15))
    
    for i, origin in enumerate(zones):
        row_sum = np.sum(matrix[i, :])
        print(f"{origin:>2}  " + "".join(f"{matrix[i, j]:>10.1f}" for j in range(len(zones))) + f"{row_sum:>10.1f}")
    
    col_sums = np.sum(matrix, axis=0)
    total_sum = np.sum(col_sums)
    print("-" * (10 * len(zones) + 15))
    print("Tot " + "".join(f"{col_sums[j]:>10.1f}" for j in range(len(zones))) + f"{total_sum:>10.1f}")

def print_probability_matrix(matrix, title, zones):
    """Print probability matrix with 3 decimal places."""
    print(f"\n{title}:")
    print("     " + "".join(f"{zone:>8}" for zone in zones))
    print("-" * (8 * len(zones) + 5))
    
    for i, origin in enumerate(zones):
        print(f"{origin:>2}  " + "".join(f"{matrix[i, j]:>8.3f}" for j in range(len(zones))))

def print_utility_matrix(matrix, title, zones):
    """Print utility matrix with 2 decimal places."""
    print(f"\n{title}:")
    print("     " + "".join(f"{zone:>8}" for zone in zones))
    print("-" * (8 * len(zones) + 5))
    
    for i, origin in enumerate(zones):
        print(f"{origin:>2}  " + "".join(f"{matrix[i, j]:>8.2f}" for j in range(len(zones))))

def main():
    """Main function to run the mode choice analysis."""
    
    print("="*80)
    print(" "*30 + "MODE CHOICE ANALYSIS")
    print(" "*25 + "Multinomial Logit Mode Choice Model")
    print("="*80)
    
    # Load OD matrix from Part 2
    od_matrix = load_od_matrix()
    
    # Setup mode data
    data = setup_mode_data()
    zones = data['zones']
    
    print("\nTotal OD Matrix from Part 2:")
    print_matrix(od_matrix, "Total Trips", zones)
    
    # Print input data
    print(f"\n" + "="*60)
    print(" "*20 + "INPUT DATA (Tables 6-12)")
    print("="*60)
    
    print_utility_matrix(data['t_walk'], "Table 6: Walking Travel Duration (minutes)", zones)
    print_utility_matrix(data['t_rail'], "Table 7: Rail Travel Duration (minutes)", zones)
    print_utility_matrix(data['t_car'], "Table 8: Car Travel Duration (minutes)", zones)
    print_utility_matrix(data['c_rail'], "Table 9: Rail Ticket Cost (£)", zones)
    print_utility_matrix(data['c_car'], "Table 10: Car Monetary Cost (£)", zones)
    print_utility_matrix(data['t_wait'], "Table 11: Rail Waiting Time (minutes)", zones)
    print_utility_matrix(data['c_park'], "Table 12: Car Parking Cost (£)", zones)
    
    # Calculate utilities
    print(f"\n" + "="*60)
    print(" "*22 + "UTILITY CALCULATIONS")
    print("="*60)
    print("\nUtility Functions:")
    print("VWALK = -0.6 * tWALK")
    print("VRAIL = 0.1 - 0.5 * tRAIL - 2 * cRAIL - 0.7 * tWAIT")
    print("VCAR  = 0.2 - 0.4 * tCAR - 1.5 * cCAR - 3 * cPARK")
    
    V_walk, V_rail, V_car = calculate_utilities(data)
    
    print_utility_matrix(V_walk, "Walking Utilities", zones)
    print_utility_matrix(V_rail, "Rail Utilities", zones)
    print_utility_matrix(V_car, "Car Utilities", zones)
    
    # Calculate mode choice probabilities
    print(f"\n" + "="*60)
    print(" "*18 + "MODE CHOICE PROBABILITIES")
    print("="*60)
    
    P_walk, P_rail, P_car = calculate_mode_probabilities(V_walk, V_rail, V_car)
    
    print_probability_matrix(P_walk, "Walking Probabilities", zones)
    print_probability_matrix(P_rail, "Rail Probabilities", zones)
    print_probability_matrix(P_car, "Car Probabilities", zones)
    
    # Calculate mode-specific OD matrices
    print(f"\n" + "="*60)
    print(" "*18 + "MODE-SPECIFIC OD MATRICES")
    print("="*60)
    
    OD_walk = od_matrix * P_walk
    OD_rail = od_matrix * P_rail
    OD_car = od_matrix * P_car
    
    print_matrix(OD_walk, "Walking OD Matrix", zones)
    print_matrix(OD_rail, "Rail OD Matrix", zones)
    print_matrix(OD_car, "Car OD Matrix", zones)
    
    # Verification - check totals
    print(f"\n" + "="*60)
    print(" "*25 + "VERIFICATION")
    print("="*60)
    
    total_walk = np.sum(OD_walk)
    total_rail = np.sum(OD_rail)
    total_car = np.sum(OD_car)
    total_all_modes = total_walk + total_rail + total_car
    original_total = np.sum(od_matrix)
    
    print(f"\nMode Split Summary:")
    print(f"Walking trips: {total_walk:,.1f} ({100*total_walk/original_total:.1f}%)")
    print(f"Rail trips:    {total_rail:,.1f} ({100*total_rail/original_total:.1f}%)")
    print(f"Car trips:     {total_car:,.1f} ({100*total_car/original_total:.1f}%)")
    print(f"Total:         {total_all_modes:,.1f}")
    print(f"Original total: {original_total:,.1f}")
    print(f"Difference:    {abs(total_all_modes - original_total):.1f}")
    
    # Save results to CSV files
    df_walk = pd.DataFrame(OD_walk, index=zones, columns=zones)
    df_rail = pd.DataFrame(OD_rail, index=zones, columns=zones)
    df_car = pd.DataFrame(OD_car, index=zones, columns=zones)
    
    df_walk.to_csv('od_matrix_walk.csv')
    df_rail.to_csv('od_matrix_rail.csv')
    df_car.to_csv('od_matrix_car.csv')
    
    # Save probabilities for reference
    df_p_walk = pd.DataFrame(P_walk, index=zones, columns=zones)
    df_p_rail = pd.DataFrame(P_rail, index=zones, columns=zones)
    df_p_car = pd.DataFrame(P_car, index=zones, columns=zones)
    
    df_p_walk.to_csv('probabilities_walk.csv')
    df_p_rail.to_csv('probabilities_rail.csv')
    df_p_car.to_csv('probabilities_car.csv')
    
    print(f"\nResults saved to CSV files:")
    print(f"- od_matrix_walk.csv")
    print(f"- od_matrix_rail.csv")
    print(f"- od_matrix_car.csv")
    print(f"- probabilities_walk.csv")
    print(f"- probabilities_rail.csv")
    print(f"- probabilities_car.csv")
    
    return OD_walk, OD_rail, OD_car

if __name__ == "__main__":
    OD_walk, OD_rail, OD_car = main()
