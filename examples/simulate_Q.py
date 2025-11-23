import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.Q_agent import QAgent

def evaluate_Q(agent, num_games=10000):
    """
    Runs games with epsilon=0 and DOES NOT update Q-values.
    This tests the agent's actual learned policy.
    """
    results = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'player_busts': 0,
        'dealer_busts': 0
    }

    print(f"\n{'='*60}")
    print(f"Simulating {num_games} games with Q-Agent")
    print(f"{'='*60}\n")

    for _ in range(num_games):
        state = agent.env.reset()
        done = False
        
        while not done:
            # Epsilon = 0 forces the agent to use its best known move (Greedy)
            action = agent.pick_action(state, epsilon=0.0) 
            
            # Step the environment
            state, reward, done, info = agent.env.step(action)
            
        # Record result after game ends
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
        else:  # draw
            results['draws'] += 1
            
    print(f"\n{'='*60}")
    print(f"STATISTICS ({num_games} games)")
    print(f"{'='*60}")
    print(f"Wins:          {results['wins']} ({results['wins']/num_games*100:.1f}%)")
    print(f"Losses:        {results['losses']} ({results['losses']/num_games*100:.1f}%)")
    print(f"Draws:         {results['draws']} ({results['draws']/num_games*100:.1f}%)")
    print(f"Player busts:  {results['player_busts']}")
    print(f"Dealer busts:  {results['dealer_busts']}")
    print(f"{'='*60}\n")

def train_evaluate_Q(num_train=50000, num_eval=10000):
    agent = QAgent()
    
    # 1. TRAINING PHASE
    print(f"Training agent ({num_train} games)...")
    agent.Q_run(num_simulation=num_train, epsilon=0.4) 
    print("Training complete.")
    
    # 2. EVALUATION PHASE
    evaluate_Q(agent, num_games=num_eval)

if __name__ == "__main__":
    train_evaluate_Q(num_train=50000, num_eval=10000)
