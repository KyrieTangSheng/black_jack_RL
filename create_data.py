from src.environment.blackjack import BlackjackEnv
from random import random
from copy import deepcopy
import json
from pathlib import Path

OUT_PATH = Path('blackjack_data.json')
N_points = 50

# Create environment
env = BlackjackEnv()

# Create dataset
dataset = []
for data_index in range(N_points):
    # Reset results and environment
    results = {}
    env.reset()

    # Update current datapoint index and dealer hand
    results['index'] = data_index
    results['dealer_hand'] = env.dealer_hand

    # Update player hands
    turn_i = 0
    while not env.game_over:
        decision = random() >= 0.5 # stand / hold with equal probability

        # add player hand and decision for that turn
        results[f'player_hand_turn{turn_i}'] = deepcopy(env.player_hand)
        results[f'player_action_turn{turn_i}'] = 'Hit' if decision else 'Stand'
        _, _, _, info = env.step(decision)
        turn_i += 1

    # add final hand and outcome
    results[f'player_hand_final'] = deepcopy(env.player_hand)
    results['outcome'] = info['result']

    dataset.append(results)

# Dump results
json.dump(dataset, OUT_PATH.open('w'), ensure_ascii=False, indent=4)