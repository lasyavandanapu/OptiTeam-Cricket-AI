import pandas as pd
import numpy as np
import os

def load_and_process_data():
    # 1. Locate the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'ipl_master_dataset.csv')
    
    print("Loading IPL master dataset...")
    df = pd.read_csv(file_path)
    
    # 2. Feature Engineering: Define the outcomes
    # If 'player_dismissed' is not empty, it's a wicket (1), otherwise (0)
    df['is_wicket'] = df['player_dismissed'].notna().astype(int)
    
    # Calculate total runs off the delivery
    df['total_runs'] = df['runs_off_bat'] + df['extras']
    
    # Group runs to avoid rare outcomes (like 5 or 7 runs) breaking the model
    # We map them to the nearest standard outcome
    df['run_outcome'] = df['total_runs'].apply(lambda x: x if x in [0, 1, 2, 3, 4, 6] else (4 if x == 5 else (6 if x > 6 else 0)))
    
    # 3. Feature Engineering: Define Match Phases
    # Extract the base over number (e.g., over 14.3 becomes 14)
    df['over'] = np.floor(df['over_ball']).astype(int)
    
    # Categorize into Powerplay (0-5), Middle (6-14), and Death (15-19)
    conditions = [
        (df['over'] < 6),
        (df['over'] >= 6) & (df['over'] < 15),
        (df['over'] >= 15)
    ]
    choices = ['Powerplay', 'Middle', 'Death']
    df['phase'] = np.select(conditions, choices, default='Middle')
    
    return df

def get_transition_probabilities():
    df = load_and_process_data()
    print("Calculating historical probabilities...")
    
    probabilities = {}
    phases = ['Powerplay', 'Middle', 'Death']
    
    for phase in phases:
        phase_data = df[df['phase'] == phase]
        total_balls = len(phase_data)
        
        # Calculate wicket probability
        total_wickets = phase_data['is_wicket'].sum()
        prob_wicket = total_wickets / total_balls if total_balls > 0 else 0
        
        # Calculate run probabilities (only for balls that are NOT wickets)
        non_wicket_data = phase_data[phase_data['is_wicket'] == 0]
        run_counts = non_wicket_data['run_outcome'].value_counts(normalize=True).to_dict()
        
        # Ensure all standard run values exist in the dictionary
        standard_runs = [0, 1, 2, 3, 4, 6]
        run_probs = {runs: run_counts.get(runs, 0.0) for runs in standard_runs}
        
        probabilities[phase] = {
            'prob_wicket': prob_wicket,
            'run_probs': run_probs
        }
        
    print("Probability Engine Ready!")
    return probabilities

# Test the script if run directly
if __name__ == "__main__":
    probs = get_transition_probabilities()
    for phase, stats in probs.items():
        print(f"\n--- {phase} Phase ---")
        print(f"Wicket Probability: {stats['prob_wicket']:.2%}")
        print("Run Probabilities (if no wicket):")
        for runs, prob in stats['run_probs'].items():
            print(f"  {runs} runs: {prob:.2%}")