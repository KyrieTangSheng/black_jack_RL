"""
Test Composition of Different Datsets on HMM Performance

We want to see if training on more wins will bias the model probabilities to achieve a better win rate.
Date: November 22, 2025
"""
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from blackjack_lib.environment.blackjack import BlackjackEnv
from helper import process_data, test_hmm
import random
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from typing import List

import matplotlib
matplotlib.use("Agg") 
from matplotlib import pyplot as plt
import json
import time

def compose_data(num_turns: List[int], N_points = 50000):

    # Create environment
    env = BlackjackEnv()

    # Create dataset
    dataset = []
    for data_index in tqdm(range(N_points)):
        while True:
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
                decision = random.random() >= 0.5 # stand / hold with equal probability
                _, _, _, info = env.step(decision)
                turn = {
                    'prev_action': 'hit' if decision else 'stand',
                    'new_card': env.player_hand[-1] if decision else None
                }
                turns.append(turn)

            # add final hand and outcome
            results['turns'] = turns
            results['outcome'] = info['result']
            if len(turns) in num_turns:
                dataset.append(results)
                break

    return dataset

# run simulation with different percentages of wins
df = []
for i, n_turns in tqdm(enumerate([[1], [2], [3], [1,2], [2,3], [1,2,3]])):
    stats = {}
    simulation = compose_data(n_turns, N_points=500000)
    random.shuffle(simulation)
    emissions, states, lengths = process_data(simulation)

    stats["n_turns"] = i
    start_time = time.perf_counter()
    winrate, drawrate = test_hmm(10000, emissions, states, lengths)
    end_time = time.perf_counter()
    stats["winrate"] = winrate
    stats["drawrate"] = drawrate
    stats['time'] = end_time - start_time
    df.append(stats)

json.dump(df, open('stats.json', 'w'), indent=4)

results = pd.DataFrame(df)
# Plot win rate vs training win percentage
plt.figure(figsize=(8, 5))
plt.plot(results["n_turns"], results["winrate"], marker="o", linewidth=2)
plt.xticks(results['n_turns'], ['1', '2', '3', '1 & 2', '2 & 3', '1 & 2 & 3'])
plt.xlabel("Training Set Num Turns")
plt.ylabel("HMM Win Rate")
plt.title("Effect of Turn Composition on HMM Performance")
plt.savefig("figures/hmm_winrate_vs_training_winpct.png")

# Optional: Plot draw rate too
plt.figure(figsize=(8, 5))
plt.plot(results["n_turns"], results["drawrate"], marker="o", linewidth=2, color="orange")
plt.xlabel("Training Set Num Turns")
plt.ylabel("HMM Draw Rate")
plt.title("Effect of Turn Composition on HMM Draw Rate")
plt.savefig("figures/hmm_drawrate_vs_training_winpct.png")