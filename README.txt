# OptiTeam: Real-Time Playbook Optimization 🏏🤖

**Data-Driven Player Selection and Dynamic Substitution Framework using Deep Reinforcement Learning**

OptiTeam is an advanced Sports Analytics and Artificial Intelligence project that transitions traditional, gut-feeling cricket strategies into a mathematically backed, real-time decision-support system. Built as an Experiential Learning Project for RV College of Engineering.

---

## 🎯 Project Overview

Traditional match strategies in cricket rely heavily on static pre-match planning and human intuition. Captains often struggle to process multidimensional live data (pitch degradation, live strike rates, bowler fatigue) during the high-pressure environment of a live match.

OptiTeam solves this by modeling a T20 cricket match as a **Markov Decision Process (MDP)**. Using a Deep Q-Network (DQN) trained on hundreds of thousands of historical IPL deliveries, the system continually assesses the live "state" of the game and outputs the highest-probability tactical actions (fielding alignments, bowling strategies) to maximize the team's chances of winning.

### Key Features
* **Custom Gymnasium Environment:** A bespoke physics engine simulating T20 cricket mechanics, utilizing real-world probability metrics.
* **Deep Reinforcement Learning (DQN):** An AI agent that actively learns the optimal balance between attacking (hunting wickets) and defending (run restriction) across different match phases.
* **FastAPI Backend Engine:** A lightweight, high-performance API that serves real-time tactical predictions.
* **3D Tactical Dashboard:** A visually immersive React/Three.js frontend allowing coaches to input live match states and view automated, spatially-rendered fielding setups.

---

## ⚙️ Tech Stack

* **Machine Learning:** `Python`, `Stable-Baselines3`, `Gymnasium`
* **Data Engineering:** `Pandas`, `NumPy`
* **Backend Server:** `FastAPI`, `Uvicorn`, `Pydantic`
* **Frontend UI:** `HTML5`, `TailwindCSS`, `Three.js` (WebGL)
* **Dataset:** `Cricsheet` (Open-Source Ball-by-Ball Data)

---

## 🚀 Installation and Setup

To run OptiTeam locally, follow these steps:

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Clone the Repository
```bash
git clone [https://github.com/YourUsername/OptiTeam-Cricket-AI.git](https://github.com/YourUsername/OptiTeam-Cricket-AI.git)
cd OptiTeam-Cricket-AI
