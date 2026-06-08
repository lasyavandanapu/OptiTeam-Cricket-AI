import gymnasium as gym
from gymnasium import spaces
import numpy as np
from data_loader import get_transition_probabilities

class CricketTacticsEnv(gym.Env):
    """Custom Environment that follows gym interface for Cricket RL"""
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(CricketTacticsEnv, self).__init__()
        
        # Load our real IPL probabilities
        self.base_probs = get_transition_probabilities()
        
        # --- ACTION SPACE ---
        # 0: Defensive Field (Tries to prevent boundaries, harder to take wickets)
        # 1: Standard Field (Normal IPL probabilities)
        # 2: Attacking Field (Tries to force wickets, high risk of conceding boundaries)
        self.action_space = spaces.Discrete(3)
        
        # --- OBSERVATION SPACE (State) ---
        # [balls_bowled (0-120), wickets_lost (0-10), total_runs_conceded (0-300)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([120, 10, 300]), 
            dtype=np.float32
        )
        
        # Initialize state variables
        self.balls_bowled = 0
        self.wickets = 0
        self.runs = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balls_bowled = 0
        self.wickets = 0
        self.runs = 0
        
        obs = np.array([self.balls_bowled, self.wickets, self.runs], dtype=np.float32)
        return obs, {}

    def step(self, action):
        # 1. Determine Match Phase
        overs = self.balls_bowled // 6
        if overs < 6:
            phase = 'Powerplay'
        elif overs < 15:
            phase = 'Middle'
        else:
            phase = 'Death'
            
        # Get base probabilities for this phase
        phase_stats = self.base_probs[phase]
        prob_w = phase_stats['prob_wicket']
        run_p = phase_stats['run_probs'].copy()
        
        # 2. Apply the Agent's Action (The Tactics)
        if action == 0: # Defensive
            prob_w *= 0.8 
            run_p[4] *= 0.7 
            run_p[6] *= 0.7 
            run_p[1] += (1 - sum(run_p.values())) 
        elif action == 2: # Attacking
            prob_w *= 1.3 
            run_p[4] *= 1.3 
            run_p[6] *= 1.4 
            
        # Normalize run probabilities so they equal 1.0 exactly
        total_run_p = sum(run_p.values())
        for k in run_p:
            run_p[k] /= total_run_p
            
        # 3. Simulate the delivery!
        reward = 0
        is_wicket_ball = np.random.random() < prob_w
        
        if is_wicket_ball:
            self.wickets += 1
            # UPDATED: Massive reward for taking a wicket to encourage attacking play!
            reward = 40 
            runs_scored = 0
        else:
            runs_scored = np.random.choice(list(run_p.keys()), p=list(run_p.values()))
            self.runs += runs_scored
            
            # UPDATED: Reward/Punishment based on runs conceded
            if runs_scored == 0:
                reward = 2 
            elif runs_scored in [1, 2, 3]:
                reward = -runs_scored 
            elif runs_scored in [4, 6]:
                # Reduced penalty so the agent isn't terrified of attacking
                reward = -runs_scored  
                
        self.balls_bowled += 1
        
        # 4. Check if game is over
        terminated = bool(self.balls_bowled >= 120 or self.wickets >= 10)
        truncated = False 
        
        # 5. End of game bonuses
        if terminated:
            if self.runs < 160: 
                reward += 50
            elif self.runs > 200: 
                reward -= 50

        obs = np.array([self.balls_bowled, self.wickets, self.runs], dtype=np.float32)
        info = {'phase': phase, 'runs_this_ball': runs_scored}
        
        return obs, reward, terminated, truncated, info