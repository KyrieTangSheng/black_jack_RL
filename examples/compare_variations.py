import sys
import os
import matplotlib.pyplot as plt

# Add path to project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack_lib.agents.Q_agent import QAgent


def run_continuous_evaluation(agent, start_game_num, num_games=10000, eval_interval=1000):
    """
    Runs evaluation (epsilon=0) and tracks BOTH Win Rate and Average Reward.
    """
    history = {
        'game_numbers': [],
        'win_rates': [],
        'rewards': []
    }

    wins = 0
    total_reward = 0
    games_count = 0

    for i in range(num_games):
        state = agent.env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Pure Greedy Strategy (No Noise)
            action = agent.pick_action(state, epsilon=0.0)
            state, reward, done, info = agent.env.step(action)
            episode_reward += reward

        if episode_reward > 0:
            wins += 1

        total_reward += episode_reward
        games_count += 1

        # Record stats at intervals
        if games_count >= eval_interval:
            current_game_idx = start_game_num + (i + 1)

            # Calculate averages for this specific window
            win_rate = wins / games_count
            avg_reward = total_reward / games_count

            history['game_numbers'].append(current_game_idx)
            history['win_rates'].append(win_rate)
            history['rewards'].append(avg_reward)

            # Reset counters
            wins = 0
            total_reward = 0
            games_count = 0

    return history


def run_full_experiment(name, max_hand_value=21, force_ace_value=None,
                        dealer_stick_threshold=17,
                        train_episodes=50000, eval_episodes=10000):
    print(f"Running {name}...", end=" ", flush=True)

    # 1. Initialize
    agent = QAgent(
        max_hand_value=max_hand_value,
        force_ace_value=force_ace_value,
        dealer_stick_threshold=dealer_stick_threshold,
        lr_base=10.0,
        discount=0.95
    )

    # 2. Train (0 to 50,000) - Epsilon 0.4
    # Using eval_interval=1000 for consistent smoothness
    agent.Q_run(num_simulation=train_episodes, epsilon=0.4,
                track_performance=True, eval_interval=1000)

    train_history = agent.training_history

    # 3. Evaluate (50,000 to 60,000) - Epsilon 0.0
    eval_history = run_continuous_evaluation(
        agent,
        start_game_num=train_episodes,
        num_games=eval_episodes,
        eval_interval=1000
    )

    print("Done.")
    return train_history, eval_history


def plot_combined_results(results_map, train_end_point=50000, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, height_ratios=[1, 1])

    # Define styles for the 3 variations
    styles = [
        {'color': 'blue', 'label': 'Baseline (21 / Dealer 17)'},
        {'color': 'orange', 'label': 'Fair Scaled (25 / Dealer 21)'},
        {'color': 'red', 'label': 'Hard Ace (21 / Ace=1)'}
    ]

    for idx, (name, (train_hist, eval_hist)) in enumerate(results_map.items()):
        style = styles[idx % len(styles)]

        # --- PREPARE DATA FOR LINKING ---
        # To link the plots, we take the LAST point of training and add it
        # as the FIRST point of evaluation arrays.

        last_train_game = train_hist['game_numbers'][-1]
        last_train_wr = train_hist['win_rates'][-1]
        last_train_rw = train_hist['rewards'][-1]

        # Create "Linked" lists for plotting
        linked_eval_games = [last_train_game] + eval_hist['game_numbers']
        linked_eval_wr = [last_train_wr] + eval_hist['win_rates']
        linked_eval_rw = [last_train_rw] + eval_hist['rewards']

        # --- SUBPLOT 1: WIN RATES ---
        ax1.plot(train_hist['game_numbers'], train_hist['win_rates'],
                 linestyle=':', linewidth=1.5, alpha=0.5, color=style['color'])

        # Plot Evaluation using the LINKED arrays
        ax1.plot(linked_eval_games, linked_eval_wr,
                 linestyle='-', linewidth=2.5, alpha=1.0, color=style['color'],
                 label=style['label'])

        # --- SUBPLOT 2: AVERAGE REWARDS ---
        ax2.plot(train_hist['game_numbers'], train_hist['rewards'],
                 linestyle=':', linewidth=1.5, alpha=0.5, color=style['color'])

        # Plot Evaluation using the LINKED arrays
        ax2.plot(linked_eval_games, linked_eval_rw,
                 linestyle='-', linewidth=2.5, alpha=1.0, color=style['color'],
                 label=style['label'])

    # --- FORMATTING ---
    ax1.axvline(x=train_end_point, color='black', linestyle='--', linewidth=1, label='Training End')
    ax1.set_ylabel('Win Rate', fontsize=12)
    ax1.set_title('Win Rate: Training vs Evaluation', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)

    ax2.axvline(x=train_end_point, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.set_xlabel('Game Number', fontsize=12)
    ax2.set_title('Average Reward: Training vs Evaluation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to {save_path}")

    plt.show()


def main():
    TRAIN_EPISODES = 50000
    EVAL_EPISODES = 10000

    results = {}

    print(f"\n{'=' * 70}")
    print(f"ABLATION STUDY (SMOOTH + LINKED PLOTS)")
    print(f"{'=' * 70}\n")

    # 1. Baseline
    name = "Baseline"
    results[name] = run_full_experiment(name, max_hand_value=21, dealer_stick_threshold=17,
                                        train_episodes=TRAIN_EPISODES, eval_episodes=EVAL_EPISODES)

    # 2. Fair Scaled
    name = "Fair Scaled"
    results[name] = run_full_experiment(name, max_hand_value=25, dealer_stick_threshold=21,
                                        train_episodes=TRAIN_EPISODES, eval_episodes=EVAL_EPISODES)

    # 3. Hard Ace
    name = "Hard Ace"
    results[name] = run_full_experiment(name, max_hand_value=21, force_ace_value=1,
                                        train_episodes=TRAIN_EPISODES, eval_episodes=EVAL_EPISODES)

    print(f"\n{'=' * 70}")
    print(f"{'EXPERIMENT':<20} | {'FINAL WIN RATE':<15} | {'AVG REWARD':<15}")
    print(f"{'-' * 70}")
    for name, (train_h, eval_h) in results.items():
        # Average of the last few evaluation points for robustness
        final_wr = sum(eval_h['win_rates'][-5:]) / 5
        final_rw = sum(eval_h['rewards'][-5:]) / 5
        print(f"{name:<20} | {final_wr:.2%}        | {final_rw:.4f}")
    print(f"{'=' * 70}\n")

    os.makedirs('figures', exist_ok=True)
    plot_combined_results(results, train_end_point=TRAIN_EPISODES,
                          save_path='figures/ablation_final_linked.png')


if __name__ == "__main__":
    main()