import gymnasium as gym
from stable_baselines3 import DQN
from cricket_env import CricketTacticsEnv

def train_model():
    print("Initializing the Cricket Environment...")
    env = CricketTacticsEnv()

    # 1. Initialize the Deep Q-Network
    print("Building the Deep Q-Network (DQN)...")
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.001, 
        buffer_size=50000, 
        exploration_fraction=0.5,
        target_update_interval=500
    )

    # 2. Train the Agent (UPDATED: Increased timesteps to 300k)
    print("\n--- Starting Training (300,000 balls) ---")
    print("This will take a few minutes. You will see logs showing its progress.")
    model.learn(total_timesteps=300000)

    # 3. Save the Brain
    model_name = "cricket_dqn_model"
    model.save(model_name)
    print(f"\nTraining Complete! Model saved as {model_name}.zip")

    return model, env

def test_model(model, env):
    print("\n--- Testing the Trained Agent in a Live Match ---")
    obs, _ = env.reset()
    
    total_reward = 0
    done = False
    
    action_names = {0: "Defensive", 1: "Standard", 2: "Attacking"}
    
    while not done:
        # The agent looks at the current state and predicts the best action
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        done = terminated or truncated
        
        # Print a few milestone overs
        balls = int(obs[0])
        if balls % 6 == 0 and balls > 0:
            over = balls // 6
            print(f"Over {over}: Agent chose {action_names[int(action)]} field | Score: {int(obs[2])}/{int(obs[1])}")

    print(f"\nMatch Finished! Final Score: {int(obs[2])}/{int(obs[1])}")
    print(f"Agent's Total Tactical Reward: {total_reward:.2f}")

if __name__ == "__main__":
    # Train it
    trained_model, test_env = train_model()
    
    # Test it
    test_model(trained_model, test_env)