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

import matplotlib
matplotlib.use("Agg") 
from matplotlib import pyplot as plt
import json
import time

def compose_data(pct_win, N_points=500000):
    """
    Simulate a dataset composition based on the given percentage of winning games.
    Args:
        pct_win (float): Percentage of winning games in the dataset (between 0 and 1).
    Returns:
        Dataset of simulated games.
    """

    # Create environment
    env = BlackjackEnv()

    # Create dataset
    dataset = []
    target_wins = int(round(pct_win * N_points))
    wins = 0
    while len(dataset) < N_points:
        # Reset results and environment
        results = {}
        env.reset()

        # Update current datapoint index and dealer hand
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

        # Define what counts as a "win"
        is_win = info["result"] in ("player_win", "dealer_bust")

        # Decide whether to keep this game based on how many wins we still need
        if is_win:
            if wins < target_wins:
                results['index'] = len(dataset)
                dataset.append(results)
                wins += 1
        else:
            # Non-wins we can keep as long as we don't exceed the non-win budget
            non_win_target = N_points - target_wins
            non_wins_so_far = len(dataset) - wins
            if non_wins_so_far < non_win_target:
                results['index'] = len(dataset)
                dataset.append(results)

    return dataset

# run simulation with different percentages of wins
df = []
for pct in tqdm([0, 0.1, 0.2, 0.3,0.4, 0.5, 0.6, 0.7]):
    stats = {}
    simulation = compose_data(pct, N_points=500000)
    random.shuffle(simulation)
    emissions, states, lengths = process_data(simulation)

    stats["pct_win"] = pct
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
plt.plot(results["pct_win"], results["winrate"], marker="o", linewidth=2)
plt.xlabel("Training Set Win Percentage (pct_win)")
plt.ylabel("HMM Win Rate")
plt.title("Effect of Training Win Composition on HMM Performance")
plt.savefig("../figures/hmm_winrate_vs_training_winpct.png")

# Optional: Plot draw rate too
plt.figure(figsize=(8, 5))
plt.plot(results["pct_win"], results["drawrate"], marker="o", linewidth=2, color="orange")
plt.xlabel("Training Set Win Percentage (pct_win)")
plt.ylabel("HMM Draw Rate")
plt.title("Effect of Training Win Composition on HMM Draw Rate")
plt.savefig("../figures/hmm_drawrate_vs_training_winpct.png")