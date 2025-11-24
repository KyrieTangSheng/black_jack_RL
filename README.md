# Blackjack Reinforcement Learning Project

CSE 150A / 250A Final Project

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Hidden Markov Model (HMM) Part

[This section is reserved for Omar & Karen's HMM part]

## Reinforcement Learning (RL) Part

This project implements Q-learning for playing Blackjack. The RL agent learns an optimal policy through training and can be evaluated on its performance.

### Running Experiments

#### 1. Hyperparameter Search

Run hyperparameter search to find the best hyperparameter combination:

```bash
python examples/hyperparameter_search.py
```

This will:
- Test different combinations of hyperparameters (discount, epsilon, learning rate)
- Display a summary table with win rates for each combination
- Identify the best hyperparameters
- Generate a detailed training/evaluation plot for the best configuration

The results are saved to `figures/q_agent_best_hyperparams.png`.

#### 2. Single Training and Evaluation

Train and evaluate a Q-agent with specific hyperparameters:

```bash
python examples/simulate_Q.py
```

This will:
- Train the agent with default hyperparameters (or modify the script to use custom values)
- Evaluate the trained agent
- Generate training and evaluation performance plots

The results are saved to `figures/q_agent_training_eval.png`.