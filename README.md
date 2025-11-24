# Blackjack Reinforcement Learning Project

CSE 150A / 250A Final Project

## Setup
From project root:
```bash
# Install dependencies
pip install -r requirements.txt
# Install blackjack_lib library
pip install -e .
```

## Hidden Markov Model (HMM) Part
Order of execution:
1. Run `python blackjack_lib/hmm/hmm_create_data.py` to create the data necessary for HMM
2. Run `python blackjack_lib/hmm/hmm_vanilla_implementation.py` to get results for HMM vanilla implementation
3. Run `python blackjack_lib/hmm/hmm_data_comparison.py` to get results for different winrate comparisons for HMM

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

### Configuring Maximum Hand Value

To change the bust threshold (default is 21), modify the constant in `blackjack_lib/environment/blackjack.py`:

```python
MAX_HAND_VALUE = 25  # Change from 21 to 25
```

This will automatically update the game rules and the Q-agent's state space to match the new threshold.