import random
from typing import List


class Deck:
    RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
    SUITS = ['♠', '♥', '♦', '♣']
    
    def __init__(self, num_decks: int = 1):
        self.num_decks = num_decks
        self.cards: List[str] = []
        self.reset()
    
    def reset(self):
        self.cards = []
        for _ in range(self.num_decks):
            for rank in self.RANKS:
                for suit in self.SUITS:
                    self.cards.append(f"{rank}{suit}")
    
    def shuffle(self):
        self.reset()
        random.shuffle(self.cards)
    
    def deal(self) -> str:
        if len(self.cards) == 0:
            self.shuffle()
        return self.cards.pop()
    
    def cards_remaining(self) -> int:
        return len(self.cards)
    
    def __len__(self) -> int:
        return len(self.cards)
    
    def __repr__(self) -> str:
        return f"Deck(num_decks={self.num_decks}, cards_remaining={len(self.cards)})"