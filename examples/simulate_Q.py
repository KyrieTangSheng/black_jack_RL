import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack_lib.agents.Q_agent import QAgent
import matplotlib.pyplot as plt

def evaluate_Q(agent, num_games=10000, track_performance=False):
    results = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'player_busts': 0,
        'dealer_busts': 0
    }
    
    eval_history = {
        'game_numbers': [],
        'win_rates': [],
        'rewards': []
    }
    
    eval_window_wins = 0
    eval_window_rewards = 0
    eval_window_games = 0
    eval_interval = max(100, num_games // 100)

    print(f"\n{'='*60}")
    print(f"Simulating {num_games} games with Q-Agent")
    print(f"{'='*60}\n")

    for game_idx in range(num_games):
        state = agent.env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.pick_action(state, epsilon=0.0) 
            state, reward, done, info = agent.env.step(action)
            episode_reward += reward
            
        result = info['result']
        if result == 'player_bust':
            results['losses'] += 1
            results['player_busts'] += 1
        elif result == 'dealer_bust':
            results['wins'] += 1
            results['dealer_busts'] += 1
        elif result == 'player_win':
            results['wins'] += 1
        elif result == 'dealer_win':
            results['losses'] += 1
        else:
            results['draws'] += 1
        
        if track_performance:
            eval_window_games += 1
            eval_window_rewards += episode_reward
            if episode_reward > 0:
                eval_window_wins += 1
            
            if eval_window_games >= eval_interval:
                win_rate = eval_window_wins / eval_window_games
                avg_reward = eval_window_rewards / eval_window_games
                
                eval_history['game_numbers'].append(game_idx + 1)
                eval_history['win_rates'].append(win_rate)
                eval_history['rewards'].append(avg_reward)
                
                eval_window_wins = 0
                eval_window_rewards = 0
                eval_window_games = 0
            
    print(f"\n{'='*60}")
    print(f"STATISTICS ({num_games} games)")
    print(f"{'='*60}")
    print(f"Wins:          {results['wins']} ({results['wins']/num_games*100:.1f}%)")
    print(f"Losses:        {results['losses']} ({results['losses']/num_games*100:.1f}%)")
    print(f"Draws:         {results['draws']} ({results['draws']/num_games*100:.1f}%)")
    print(f"Player busts:  {results['player_busts']}")
    print(f"Dealer busts:  {results['dealer_busts']}")
    print(f"{'='*60}\n")
    
    if track_performance:
        return eval_history
    return None

def plot_training_evaluation_performance(agent, eval_history, num_train, num_eval, epsilon, 
                                         discount, train_eval_interval, save_path=None, lr_base=10.0):
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.15], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax_text = fig.add_subplot(gs[2])
    ax_text.axis('off')
    
    if agent.training_history['game_numbers']:
        train_games = agent.training_history['game_numbers']
        train_win_rates = agent.training_history['win_rates']
        ax1.plot(train_games, train_win_rates, 'b-', label='Training', linewidth=2, alpha=0.7)
    
    if eval_history and eval_history['game_numbers']:
        eval_games = [num_train + g for g in eval_history['game_numbers']]
        eval_win_rates = eval_history['win_rates']
        ax1.plot(eval_games, eval_win_rates, 'r-', label='Evaluation', linewidth=2, alpha=0.7)
    
    ax1.axvline(x=num_train, color='green', linestyle='--', linewidth=2, 
                label=f'Training End (Game {num_train})', alpha=0.8)
    
    ax1.set_xlabel('Game Number', fontsize=12)
    ax1.set_ylabel('Win Rate', fontsize=12)
    ax1.set_title('Win Rate During Training and Evaluation', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    if agent.training_history['game_numbers']:
        train_games = agent.training_history['game_numbers']
        train_rewards = agent.training_history['rewards']
        ax2.plot(train_games, train_rewards, 'b-', label='Training', linewidth=2, alpha=0.7)
    
    if eval_history and eval_history['game_numbers']:
        eval_games = [num_train + g for g in eval_history['game_numbers']]
        eval_rewards = eval_history['rewards']
        ax2.plot(eval_games, eval_rewards, 'r-', label='Evaluation', linewidth=2, alpha=0.7)
    
    ax2.axvline(x=num_train, color='green', linestyle='--', linewidth=2, 
                label=f'Training End (Game {num_train})', alpha=0.8)
    
    ax2.set_xlabel('Game Number', fontsize=12)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.set_title('Average Reward During Training and Evaluation', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    hyperparams_text = (
        f"Hyperparameters:  "
        f"Discount (γ) = {discount:.2f}  |  "
        f"Epsilon (ε) = {epsilon:.2f}  |  "
        f"Learning Rate α(n) = {lr_base:.1f}/(9+n)  |  "
        f"Training Games = {num_train:,}  |  "
        f"Evaluation Games = {num_eval:,}  |  "
        f"Training Eval Interval = {train_eval_interval:,}"
    )
    
    ax_text.text(0.5, 0.5, hyperparams_text, 
                fontsize=11, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                family='monospace')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

def evaluate_win_rate(agent, num_games=10000):
    wins = 0
    for _ in range(num_games):
        state = agent.env.reset()
        done = False
        while not done:
            action = agent.pick_action(state, epsilon=0.0)
            state, reward, done, info = agent.env.step(action)
        if reward > 0:
            wins += 1
    return wins / num_games

def train_evaluate_Q(num_train=50000, num_eval=10000, track_performance=True, 
                     train_eval_interval=1000, epsilon=0.4, discount=0.95, lr_base=10.0,
                     plot=True, save_plot=None):
    agent = QAgent(discount=discount, lr_base=lr_base)
    
    print(f"Training agent ({num_train} games)...")
    train_start = time.time()
    agent.Q_run(num_simulation=num_train, epsilon=epsilon, 
                track_performance=track_performance, 
                eval_interval=train_eval_interval) 
    train_time = time.time() - train_start
    print(f"Training complete. Time: {train_time:.2f}s ({train_time/60:.2f} minutes)")
    
    eval_start = time.time()
    eval_history = evaluate_Q(agent, num_games=num_eval, track_performance=track_performance)
    eval_time = time.time() - eval_start
    print(f"Evaluation complete. Time: {eval_time:.2f}s")
    print(f"Total time: {train_time + eval_time:.2f}s ({(train_time + eval_time)/60:.2f} minutes)\n")
    
    if plot and track_performance:
        plot_training_evaluation_performance(agent, eval_history, num_train, num_eval, 
                                            epsilon, discount, train_eval_interval, save_plot,
                                            lr_base=lr_base)
    
    return agent, eval_history

if __name__ == "__main__":
    train_evaluate_Q(num_train=50000, num_eval=10000, 
                     track_performance=True, 
                     train_eval_interval=1000,
                     epsilon=0.4,
                     plot=True,
                     save_plot='figures/q_agent_training_eval.png')
