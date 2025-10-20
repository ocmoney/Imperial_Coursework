"""
Zone Trip Generation Calculator - Simple version that just calculates total trips by zone.
"""

import pandas as pd
from regression import load_data, create_dummy_variables, estimate_trip_generation_model

# Load model and data
data = load_data('Data_CW1.csv')
data = create_dummy_variables(data)
model, X, y = estimate_trip_generation_model(data)

# Get segment-specific average household sizes
#calculate average household sizes: use only segment specific average household sizes, and not zone specific average household sizes for each segment. 
segment_hh_sizes = data.groupby('Segment')['HH Size'].mean().to_dict() 

# Projected households by zone and segment (Table 2)
projected = {
    'Zone': ['A', 'B', 'C', 'D', 'E'],
    'Seg1': [200, 120, 80, 70, 80], 'Seg2': [120, 90, 70, 80, 130], 'Seg3': [80, 70, 100, 120, 180],
    'Seg4': [60, 80, 90, 130, 200], 'Seg5': [70, 90, 150, 200, 280], 'Seg6': [90, 150, 200, 270, 370],
    'Seg7': [170, 240, 350, 460, 600], 'Seg8': [240, 350, 450, 620, 770], 'Seg9': [380, 450, 630, 750, 900]
}
projected_df = pd.DataFrame(projected)

# Segment dummy variable mappings
segment_dummies = {
    1: [0, 0, 0, 0, 0, 0], 2: [0, 0, 1, 0, 0, 0], 3: [0, 0, 0, 0, 0, 1], 4: [1, 0, 0, 0, 0, 0],
    5: [1, 0, 0, 1, 0, 0], 6: [1, 0, 0, 0, 0, 1], 7: [0, 1, 0, 0, 0, 0], 8: [0, 1, 0, 0, 1, 0], 9: [0, 1, 0, 0, 0, 1]
}

# Calculate trips for each zone
print("TOTAL TRIPS BY ZONE")
print("=" * 40)
print(f"{'Zone':<6} {'Total Trips':<12}")
print("-" * 20)

total_all_zones = 0

for _, row in projected_df.iterrows():
    zone = row['Zone']
    zone_total = 0
    
    for seg in range(1, 10):
        households = row[f'Seg{seg}']
        hh_size = segment_hh_sizes[seg] 
        
        # Calculate trips per household using regression model
        dummies = segment_dummies[seg]  # [MedInc, HigInc, 1CarLowInc, 1CarMedInc, 1CarHigInc, 2+Cars]
        trips_per_hh = (model.params['const'] + 
                       model.params['MedInc'] * dummies[0] + model.params['HigInc'] * dummies[1] +
                       model.params['1CarLowInc'] * dummies[2] + model.params['1CarMedInc'] * dummies[3] +
                       model.params['1CarHigInc'] * dummies[4] + model.params['2+Cars'] * dummies[5] +
                       model.params['HH Size'] * hh_size)
        
        zone_total += households * trips_per_hh
    
    total_all_zones += zone_total
    print(f"{zone:<6} {zone_total:<12,.0f}")

print("-" * 20)
print(f"{'TOTAL':<6} {total_all_zones:<12,.0f}")
print("=" * 40)
