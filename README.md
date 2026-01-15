# SnakeAI with Deep Q-Learning (DQN) 🐍🧠

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Arcade](https://img.shields.io/badge/Library-Arcade-brightgreen.svg)

An autonomous agent that learns to play the classic Snake game using **Deep Reinforcement Learning**.
This project demonstrates the implementation of a **Deep Q-Network (DQN)** from scratch using PyTorch, applied to a custom game environment built with the Python `arcade` library.

## 🎯 Objective & Theory

The goal is to train a neural network to approximate the Q-value function $Q(s, a)$, which represents the maximum expected future reward for taking action $a$ in state $s$.

### Mathematical Foundation: The Bellman Equation
The agent updates its policy using the **Bellman Optimality Equation**:

$$Q_{new}(s, a) = r + \gamma \max_{a'} Q(s', a')$$

Where:
- $r$ is the immediate reward (Food: +10, Collision: -10, Else: 0).
- $\gamma$ is the discount factor (set to 0.9).
- $s'$ is the next state.

### Architecture
- **Input Layer (11 neurons):** Perception of the environment (dangers straight/left/right, current direction, food location).
- **Hidden Layer (256 neurons):** Dense layer with ReLU activation.
- **Output Layer (3 neurons):** Action space (Straight, Right Turn, Left Turn).
- **Optimizer:** Adam with Mean Squared Error (MSE) loss.

## 🛠️ Tech Stack & Features

* **PyTorch:** For building the Linear QNet and handling tensor operations.
* **Python Arcade:** A modern, object-oriented Python library used for the game engine (rendering and physics).
* **Experience Replay:** Implemented a `deque` memory buffer to store past transitions $(s, a, r, s', done)$ and train on random mini-batches, breaking correlation between consecutive steps and stabilizing training.
* **Epsilon-Greedy Strategy:** Balances exploration (random moves) and exploitation (model prediction) during the training phase.

## 🚀 Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/hgbekrar/SnakeAI.git](https://github.com/hgbekrar/SnakeAI.git)
    cd SnakeAI
    ```

2.  **Install dependencies**
    ```bash
    pip install torch arcade numpy matplotlib
    ```

3.  **Run the Training Agent**
    ```bash
    python agent.py
    ```
    *The agent will start training immediately. The model is saved automatically to `./model/model.pth` whenever a new high score is reached.*

## 📈 Results
The agent typically starts showing avoiding behaviors after 50 games and begins to actively seek food effectively after 100-150 games.
