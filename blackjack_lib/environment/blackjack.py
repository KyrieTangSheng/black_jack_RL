from typing import Tuple, List, Dict
from .deck import Deck

MAX_HAND_VALUE = 25


class BlackjackEnv:
    """Blackjack environment for reinforcement learning."""
    
    def __init__(self, num_decks: int = 1, natural_payout: float = 1.0, max_hand_value: int = MAX_HAND_VALUE):
        self.num_decks = num_decks
        self.natural_payout = natural_payout
        self.max_hand_value = max_hand_value
        self.deck = Deck(num_decks=num_decks)
        
        self.player_hand: List[str] = []
        self.dealer_hand: List[str] = []
        self.player_sum = 0
        self.dealer_sum = 0
        self.usable_ace = False
        self.game_over = False
        
    def reset(self) -> Tuple[int, int, bool]:
        self.deck.shuffle()
        self.game_over = False
        
        self.player_hand = [self.deck.deal(), self.deck.deal()]
        self.dealer_hand = [self.deck.deal(), self.deck.deal()]
        
        self.player_sum, self.usable_ace = self._calculate_hand(self.player_hand)
        self.dealer_sum, _ = self._calculate_hand(self.dealer_hand)
        
        dealer_card = self._card_value(self.dealer_hand[0])
        return (self.player_sum, dealer_card, self.usable_ace)
    
    def step(self, action: int) -> Tuple[Tuple[int, int, bool], float, bool, Dict]:
        if self.game_over:
            raise Exception("Game is over. Call reset() to start a new game.")
        
        dealer_card = self._card_value(self.dealer_hand[0])
        
        if action == 1:
            self.player_hand.append(self.deck.deal())
            self.player_sum, self.usable_ace = self._calculate_hand(self.player_hand)
            
            if self.player_sum > self.max_hand_value:
                self.game_over = True
                return (self.player_sum, dealer_card, self.usable_ace), -1.0, True, {'result': 'player_bust'}
            
            return (self.player_sum, dealer_card, self.usable_ace), 0.0, False, {}
        else:
            reward, result = self._dealer_play()
            self.game_over = True
            return (self.player_sum, dealer_card, self.usable_ace), reward, True, {'result': result}
    
    def _dealer_play(self) -> Tuple[float, str]:
        self.dealer_sum, _ = self._calculate_hand(self.dealer_hand)
        
        while self.dealer_sum < 17:
            self.dealer_hand.append(self.deck.deal())
            self.dealer_sum, _ = self._calculate_hand(self.dealer_hand)
        
        if self.dealer_sum > self.max_hand_value:
            return 1.0, 'dealer_bust'
        elif self.player_sum > self.dealer_sum:
            return 1.0, 'player_win'
        elif self.player_sum < self.dealer_sum:
            return -1.0, 'dealer_win'
        else:
            return 0.0, 'draw'
    
    def _calculate_hand(self, hand: List[str]) -> Tuple[int, bool]:
        hand_sum = sum(self._card_value(card) for card in hand)
        usable_ace = False
        
        num_aces = sum(1 for card in hand if card[0] == 'A')
        
        if num_aces > 0 and hand_sum + 10 <= self.max_hand_value:
            hand_sum += 10
            usable_ace = True
        
        return hand_sum, usable_ace
    
    def _card_value(self, card: str) -> int:
        rank = card[0]
        if rank == 'A':
            return 1
        elif rank in ['J', 'Q', 'K']:
            return 10
        elif rank == '1':
            return 10
        else:
            return int(rank)
    
    def render(self, show_dealer_card: bool = True):
        print("\n" + "="*50)
        print(f"Player hand: {' '.join(self.player_hand)}")
        print(f"Player sum: {self.player_sum} (usable ace: {self.usable_ace})")
        print("-"*50)
        
        if show_dealer_card or self.game_over:
            print(f"Dealer hand: {' '.join(self.dealer_hand)}")
            print(f"Dealer sum: {self.dealer_sum}")
        else:
            print(f"Dealer hand: {self.dealer_hand[0]} [Hidden]")
            print(f"Dealer showing: {self._card_value(self.dealer_hand[0])}")
        
        print("="*50 + "\n")
    
    def get_state(self) -> Tuple[int, int, bool]:
        dealer_card = self._card_value(self.dealer_hand[0])
        return (self.player_sum, dealer_card, self.usable_ace)


class InteractiveBlackjack(BlackjackEnv):
    def play_interactive(self):
        print("\nWelcome to Blackjack!")
        print(f"Goal: Get closer to {self.max_hand_value} than the dealer without going over!")
        print("Actions: 'h' = Hit (take a card), 's' = Stand (stop)")
        
        state = self.reset()
        self.render(show_dealer_card=False)
        
        while not self.game_over:
            action_str = input("Your action (h/s): ").lower().strip()
            
            if action_str not in ['h', 's']:
                print("Invalid input! Please enter 'h' for Hit or 's' for Stand.")
                continue
            
            action = 1 if action_str == 'h' else 0
            next_state, reward, done, info = self.step(action)
            
            if action == 1:
                self.render(show_dealer_card=False)
            
            if done:
                self.render(show_dealer_card=True)
                
                result = info['result']
                if result == 'player_bust':
                    print(f"BUST! You went over {self.max_hand_value}. Dealer wins!")
                elif result == 'dealer_bust':
                    print("Dealer busts! You win!")
                elif result == 'player_win':
                    print("You win!")
                elif result == 'dealer_win':
                    print("Dealer wins!")
                else:
                    print("It's a draw!")
                
                print(f"Reward: {reward:+.0f}")
                break
        
        play_again = input("\nPlay again? (y/n): ").lower().strip()
        if play_again == 'y':
            self.play_interactive()