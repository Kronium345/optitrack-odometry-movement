import pandas as pd

# Load the CSV file as a plain text file to inspect its structure
with open('slam_odometry_20240717.csv', 'r') as file:
    lines = file.readlines()

# Display the first few lines to understand its structure
for line in lines[:20]:
    print(line.strip())

# Function to extract relevant fields from each entry
def parse_odometry_data(lines):
    data = []
    entry = {}
    for line in lines:
        line = line.strip()
        if line.startswith('sec:'):
            entry['sec'] = float(line.split('sec:')[1].strip())
        elif line.startswith('nanosec:'):
            entry['nanosec'] = float(line.split('nanosec:')[1].strip()) * 1e-9
        elif line.startswith('position:'):
            entry['position'] = True
        elif line.startswith('x:') and 'position' in entry:
            entry['pos_x'] = float(line.split('x:')[1].strip())
        elif line.startswith('y:') and 'position' in entry:
            entry['pos_y'] = float(line.split('y:')[1].strip())
        elif line.startswith('z:') and 'position' in entry:
            entry['pos_z'] = float(line.split('z:')[1].strip())
        elif line.startswith('---'):
            if 'position' in entry:
                entry['sec'] += entry['nanosec']
                data.append(entry)
                print(f"Appended entry: {entry}")
            entry = {}
        elif 'position' in entry and not line.startswith(('x:', 'y:', 'z:')):
            entry.pop('position', None)
    return data

# Parse the odometry data
parsed_data = parse_odometry_data(lines)

# Debugging: Print parsed data
for entry in parsed_data[:5]:
    print(entry)

# Convert to DataFrame
odometry_df = pd.DataFrame(parsed_data)

# Save the restructured data to a new CSV file
odometry_df.to_csv('restructured_slam_odometry_20240717.csv', index=False)

# Display the first few rows of the restructured DataFrame
print(odometry_df.head())
