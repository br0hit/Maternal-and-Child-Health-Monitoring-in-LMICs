import pandas as pd
from scipy.spatial import distance

# Load the training_label.csv file
data = pd.read_csv('../data/training_label.csv', converters={'LATNUM': float, 'LONGNUM': float})

print(data.columns)

# Calculate mean and median values for each region
region_stats = data.groupby(data['DHSID'].str[:2]).agg({'Mean_BMI': 'mean', 'Median_BMI': 'median'})

# Function to calculate distance between two coordinates
def calculate_distance(lat1, lon1, lat2, lon2):
    return distance.haversine((lat1, lon1), (lat2, lon2))

def find_nearest_regions(row, known_regions, k=5):
    distances = []
    for _, known_region in known_regions.iterrows():
        dist = calculate_distance(row['LATNUM'], row['LONGNUM'], known_region['LATNUM'], known_region['LONGNUM'])
        distances.append((known_region.name, dist))
    distances.sort(key=lambda x: x[1])  # Sort distances in ascending order
    nearest_regions = [x[0] for x in distances[:k]]
    return nearest_regions

# Create a copy of the original data
filled_data = data.copy()

# Fill missing Mean_BMI and Median_BMI values
for index, row in filled_data[filled_data['Mean_BMI'].isnull() | filled_data['Median_BMI'].isnull()].iterrows():
    region = row['DHSID'][:2]
    if region in region_stats.index:
        
        known_regions = region_stats[region_stats.index != region]
        nearest_regions = find_nearest_regions(row, known_regions)
        
        weights = [1 / (dist + 1) for dist in range(len(nearest_regions))]
        weighted_mean = sum(region_stats.loc[nearest_regions, 'Mean_BMI'] * weights) / sum(weights)
        weighted_median = sum(region_stats.loc[nearest_regions, 'Median_BMI'] * weights) / sum(weights)
        if pd.isnull(row['Mean_BMI']):
            filled_data.at[index, 'Mean_BMI'] = weighted_mean
        if pd.isnull(row['Median_BMI']):
            filled_data.at[index, 'Median_BMI'] = weighted_median

# Save the filled data to a new CSV file
filled_data.to_csv('../data/filled_data.csv', index=False)

# Now you have a new CSV file named 'filled_data.csv' with filled missing values for Mean_BMI and Median_BMI
