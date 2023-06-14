import csv
import random

# Path to the input CSV file
input_file = '../data/gee_features.csv'

# Path to the output CSV file for the sampled data
output_file = 'sampled_data.csv'

# Number of rows to randomly select
sample_size = 20000

# Read the header row from the input file
with open(input_file, 'r') as file:
    reader = csv.reader(file)
    header = next(reader)

# Randomly select rows and write them to the output file
with open(input_file, 'r') as file, open(output_file, 'w', newline='') as output:
    reader = csv.reader(file)
    writer = csv.writer(output)

    # Write the header row to the output file
    writer.writerow(header)

    # Skip the header row for random selection
    rows = list(reader)[1:]

    # Randomly select sample_size rows
    random_sample = random.sample(rows, sample_size)

    # Write the randomly selected rows to the output file
    writer.writerows(random_sample)

print(f"Randomly sampled {sample_size} rows and saved to '{output_file}'.")
