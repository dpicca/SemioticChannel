"""
Target Concepts Dataset for the Taboo Semiotic Experiment.
Contains 50 concrete and 50 abstract concepts.
"""

import os

CONCRETE_CONCEPTS = [
    # Objects
    "Clock", "Bicycle", "Umbrella", "Mirror", "Candle",
    "Hammer", "Telescope", "Piano", "Scissors", "Compass",
    # Nature
    "Mountain", "River", "Forest", "Desert", "Ocean",
    "Volcano", "Glacier", "Waterfall", "Cave", "Island",
    # Animals
    "Elephant", "Butterfly", "Owl", "Dolphin", "Spider",
    "Eagle", "Whale", "Fox", "Turtle", "Bee",
    # Food
    "Bread", "Apple", "Honey", "Coffee", "Cheese",
    "Wine", "Chocolate", "Salt", "Pepper", "Lemon",
    # Architecture
    "Bridge", "Tower", "Castle", "Lighthouse", "Temple",
    "Fountain", "Statue", "Gate", "Wall", "Stairs",
]

ABSTRACT_CONCEPTS = [
    # Emotions
    "Nostalgia", "Joy", "Fear", "Hope", "Grief",
    "Love", "Anger", "Serenity", "Anxiety", "Pride",
    # Values
    "Justice", "Freedom", "Honor", "Courage", "Wisdom",
    "Truth", "Loyalty", "Humility", "Integrity", "Mercy",
    # Philosophy
    "Time", "Infinity", "Existence", "Consciousness", "Fate",
    "Chaos", "Order", "Paradox", "Entropy", "Eternity",
    # Social
    "Democracy", "Revolution", "Tradition", "Progress", "Identity",
    "Power", "Equality", "Culture", "Legacy", "Rebellion",
    # Abstract States
    "Silence", "Solitude", "Mystery", "Harmony", "Tension",
    "Balance", "Rhythm", "Irony", "Absurdity", "Beauty",
]

ALL_CONCEPTS = CONCRETE_CONCEPTS + ABSTRACT_CONCEPTS

def get_concepts(category: str = "all", n: int = None, file_path: str = None) -> list[str]:
    """
    Returns concepts. Can load from file if file_path (str) is provided.
    
    Args:
        category: "concrete", "abstract", or "all"
        n: Optional limit on number of concepts to return
        file_path: Optional path to external text file (one concept per line)
    
    Returns:
        List of concept strings
    """
    concepts = []
    
    # Try loading from file if provided
    if file_path:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    concepts = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"Error reading concepts file: {e}")
        else:
            print(f"Warning: Concepts file '{file_path}' not found.")
            
    # Fallback to hardcoded lists if no file or loading failed
    if not concepts:
        if category == "concrete":
            concepts = CONCRETE_CONCEPTS.copy()
        elif category == "abstract":
            concepts = ABSTRACT_CONCEPTS.copy()
        else:
            concepts = ALL_CONCEPTS.copy()
    
    if n is not None and n > 0 and n < len(concepts):
        # Sample randomly or take first n? Original took first n.
        return concepts[:n]
    return concepts

def get_concept_category(concept: str) -> str:
    """Returns whether a concept is 'concrete' or 'abstract'."""
    if concept in CONCRETE_CONCEPTS:
        return "concrete"
    elif concept in ABSTRACT_CONCEPTS:
        return "abstract"
    return "unknown"
