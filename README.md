# Blackjack Reinforcement Learning Project

CSE 150A / 250A Final Project

### Setup
```bash
# Install dependencies
pip install numpy
```

### File Structure (So Far)
```
blackjack-rl-project/
├── src/
│   └── environment/
│       ├── __init__.py
│       ├── deck.py          # Card deck implementation
│       └── blackjack.py     # Game environment
└── examples/
    └── simulate_game.py     # An Example Test/simulation script
```

### Testing the Game

Run the simulation script:
```bash
python examples/simulate_game.py
```

This will give you an interactive menu to:
1. Test environment functionality
2. Simulate random games
3. Play with basic strategy
4. Play interactively as a human player

### Quick Test in Python

```python
from src.environment.blackjack import BlackjackEnv

# Create environment
env = BlackjackEnv()

# Start a game
state = env.reset()
print(f"Initial state: {state}")  # (player_sum, dealer_card, usable_ace)

# Take action (0=stand, 1=hit)
next_state, reward, done, info = env.step(1)  # Hit

# Display the game
env.render()
```

## Game Rules

- **Goal**: Get closer to 21 than the dealer without going over
- **Actions**: 
  - Hit (1): Take another card
  - Stand (0): Stop taking cards
- **Dealer Strategy**: Hits on 16 or below, stands on 17+
- **Card Values**: 
  - Number cards: Face value
  - Face cards (J, Q, K): 10
  - Ace: 1 or 11 (whichever is better)