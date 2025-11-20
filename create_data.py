from src.environment.blackjack import BlackjackEnv
from random import random
from copy import deepcopy
import json
from pathlib import Path
from tqdm import tqdm

OUT_PATH = Path('blackjack_data.json')
N_points = 500000

# Create environment
env = BlackjackEnv()

# Create dataset
dataset = []
for data_index in tqdm(range(N_points)):
    # Reset results and environment
    results = {}
    env.reset()

    # Update current datapoint index and dealer hand
    results['index'] = data_index
    results['dealer_hand'] = env.dealer_hand
    results['player_hand'] = deepcopy(env.player_hand)

    # Update player hands
    turns = []
    while not env.game_over:
        decision = random() >= 0.5 # stand / hold with equal probability
        _, _, _, info = env.step(decision)
        turn = {
            'prev_action': 'hit' if decision else 'stand',
            'new_card': env.player_hand[-1] if decision else None
        }
        turns.append(turn)

    # add final hand and outcome
    results['turns'] = turns
    results['outcome'] = info['result']

    dataset.append(results)

# Dump results
json.dump(dataset, OUT_PATH.open('w'), ensure_ascii=False)