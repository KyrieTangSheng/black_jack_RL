import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blackjack_lib.environment.blackjack import BlackjackEnv, InteractiveBlackjack
import random


def simulate_random_games(num_games: int = 10, verbose: bool = True):
    env = BlackjackEnv()
    
    results = {
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'player_busts': 0,
        'dealer_busts': 0
    }
    
    print(f"\n{'='*60}")
    print(f"Simulating {num_games} games with RANDOM actions")
    print(f"{'='*60}\n")
    
    for game in range(num_games):
        state = env.reset()
        
        if verbose:
            print(f"\n--- Game {game + 1} ---")
            env.render(show_dealer_card=False)
        
        done = False
        total_reward = 0
        
        while not done:
            player_sum, dealer_card, usable_ace = state
            action = 1 if player_sum < 17 else 0
            
            if verbose:
                action_str = "HIT" if action == 1 else "STAND"
                print(f"Action: {action_str}")
            
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            
            if verbose and action == 1 and not done:
                env.render(show_dealer_card=False)
        
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
        
        if verbose:
            env.render(show_dealer_card=True)
            print(f"Result: {result.upper()} | Reward: {total_reward:+.0f}")
    
    print(f"\n{'='*60}")
    print(f"STATISTICS ({num_games} games)")
    print(f"{'='*60}")
    print(f"Wins:          {results['wins']} ({results['wins']/num_games*100:.1f}%)")
    print(f"Losses:        {results['losses']} ({results['losses']/num_games*100:.1f}%)")
    print(f"Draws:         {results['draws']} ({results['draws']/num_games*100:.1f}%)")
    print(f"Player busts:  {results['player_busts']}")
    print(f"Dealer busts:  {results['dealer_busts']}")
    print(f"{'='*60}\n")
    
    return results


def simulate_basic_strategy_game(verbose: bool = True):
    env = BlackjackEnv()
    state = env.reset()
    
    if verbose:
        print(f"\n{'='*60}")
        print("Playing with BASIC STRATEGY (hit on <17, stand on ≥17)")
        print(f"{'='*60}")
        env.render(show_dealer_card=False)
    
    done = False
    total_reward = 0
    
    while not done:
        player_sum, dealer_card, usable_ace = state
        
        if player_sum < 17:
            action = 1
            action_str = "HIT"
        else:
            action = 0
            action_str = "STAND"
        
        if verbose:
            print(f"Player sum: {player_sum} → Action: {action_str}")
        
        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        
        if verbose and action == 1 and not done:
            env.render(show_dealer_card=False)
    
    if verbose:
        env.render(show_dealer_card=True)
        result = info['result']
        print(f"Result: {result.upper()} | Reward: {total_reward:+.0f}")
        print(f"{'='*60}\n")
    
    return total_reward, info['result']


def test_environment():
    print("\nTesting Environment Functionality...\n")
    
    env = BlackjackEnv()
    
    print("Test 1: Environment reset")
    state = env.reset()
    print(f"  Initial state: {state}")
    assert len(state) == 3, "State should be (player_sum, dealer_card, usable_ace)"
    
    print("Test 2: Player hits")
    next_state, reward, done, info = env.step(1)
    print(f"  After hit: {next_state}, done={done}")
    
    print("Test 3: Stand immediately")
    env.reset()
    next_state, reward, done, info = env.step(0)
    print(f"  After stand: reward={reward}, done={done}, result={info['result']}")
    
    print("\nAll tests passed!\n")


def main():
    print("\n" + "="*60)
    print("BLACKJACK ENVIRONMENT SIMULATOR")
    print("="*60)
    
    while True:
        print("\nChoose an option:")
        print("1. Test environment functionality")
        print("2. Simulate 10 random games (verbose)")
        print("3. Simulate 1000 games (statistics only)")
        print("4. Play one game with basic strategy")
        print("5. Play interactively (human player)")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            test_environment()
        
        elif choice == '2':
            simulate_random_games(num_games=10, verbose=True)
        
        elif choice == '3':
            simulate_random_games(num_games=1000, verbose=False)
        
        elif choice == '4':
            simulate_basic_strategy_game(verbose=True)
        
        elif choice == '5':
            game = InteractiveBlackjack()
            game.play_interactive()
        
        elif choice == '6':
            print("\nThanks for playing! Goodbye!\n")
            break
        
        else:
            print("Invalid choice. Please enter 1-6.")


if __name__ == '__main__':
    main()