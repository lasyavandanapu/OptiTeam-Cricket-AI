from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import DQN

# 1. Initialize the FastAPI app
app = FastAPI(title="Cricket AI Tactician API")

# Allow the frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the trained AI Brain
print("Loading AI Brain...")
try:
    model = DQN.load("cricket_dqn_model.zip")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# 3. Define the data format the frontend will send us
class MatchState(BaseModel):
    overs_bowled: float
    wickets_lost: int
    runs_conceded: int

# 4. Create the Prediction Endpoint
@app.post("/predict-strategy")
async def predict_strategy(state: MatchState):
    # Convert incoming data into the format the Gym Environment expects
    obs = np.array([state.overs_bowled * 6, state.wickets_lost, state.runs_conceded], dtype=np.float32)
    
    # Ask the AI for the best action
    action, _ = model.predict(obs, deterministic=True)
    action_id = int(action)
    
    # Map the numerical action back to human-readable strategies
    strategies = {
        0: {
            "name": "Defensive Field",
            "description": "Push fielders to the boundary. Focus on bowling wide lines to restrict boundaries. Accept that taking a wicket is unlikely.",
            "risk_level": "Low"
        },
        1: {
            "name": "Standard Field",
            "description": "Keep a balanced field. Bowl tight lines on the stumps. Wait for the batsman to make a mistake.",
            "risk_level": "Medium"
        },
        2: {
            "name": "Attacking Field",
            "description": "Bring fielders into the ring. Bowl aggressive lengths (yorkers/bouncers) targeting the stumps/body. High risk of conceding boundaries, but high probability of a wicket.",
            "risk_level": "High"
        }
    }
    
    recommendation = strategies[action_id]
    
    return {
        "action_id": action_id,
        "recommendation": recommendation
    }

# Run this server by typing: uvicorn main:app --reload