import pandas as pd
import glob
import os
import csv

# Get the directory where this script is saved
script_dir = os.path.dirname(os.path.abspath(__file__))
file_pattern = os.path.join(script_dir, '*.csv')
file_list = glob.glob(file_pattern)

# Filter for files that are purely numbers (match files)
match_files = [f for f in file_list if os.path.basename(f).split('.')[0].isdigit()]

print(f"Found {len(match_files)} IPL match files.")
print("Extracting ball-by-ball data... this will take a moment.")

all_deliveries = []

# Standard columns for a Cricsheet 'ball' row
columns = [
    'match_id', 'innings', 'over_ball', 'batting_team', 'striker', 
    'non_striker', 'bowler', 'runs_off_bat', 'extras', 
    'wicket_type', 'player_dismissed'
]

for file in match_files:
    match_id = os.path.basename(file).split('.')[0]
    
    with open(file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # We ONLY want rows that start with the word 'ball'
            if len(row) > 0 and row[0] == 'ball':
                # The row looks like: ['ball', '1', '0.1', 'CSK', 'MS Dhoni', ...]
                # We drop the word 'ball' (index 0) and insert our match_id instead
                clean_row = [match_id] + row[1:]
                
                # Sometimes rows are shorter if there is no wicket. 
                # We pad the row with empty strings to ensure it matches our 11 columns
                while len(clean_row) < len(columns):
                    clean_row.append('')
                    
                # If the row somehow has MORE than 11 columns (extra fielder data), truncate it
                clean_row = clean_row[:11]
                
                all_deliveries.append(clean_row)

print("Formatting data into a clean table...")

# Convert our extracted data into a standard pandas DataFrame
df = pd.DataFrame(all_deliveries, columns=columns)

# Save the master file
output_name = os.path.join(script_dir, 'ipl_master_dataset.csv')
df.to_csv(output_name, index=False)

print("--------------------------------------------------")
print(f"Success! Master dataset saved to:\n{output_name}")
print(f"Total deliveries extracted for your AI: {len(df)}")
print("--------------------------------------------------")