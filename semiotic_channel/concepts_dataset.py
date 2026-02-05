"""
Target Concepts Dataset for the Taboo Semiotic Experiment.
Contains 50 concrete and 50 abstract concepts.
"""

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

def get_concepts(category: str = "all", n: int = None) -> list[str]:
    """
    Returns concepts by category.
    
    Args:
        category: "concrete", "abstract", or "all"
        n: Optional limit on number of concepts to return
    
    Returns:
        List of concept strings
    """
    if category == "concrete":
        concepts = CONCRETE_CONCEPTS.copy()
    elif category == "abstract":
        concepts = ABSTRACT_CONCEPTS.copy()
    else:
        concepts = ALL_CONCEPTS.copy()
    
    if n is not None and n < len(concepts):
        return concepts[:n]
    return concepts

def get_concept_category(concept: str) -> str:
    """Returns whether a concept is 'concrete' or 'abstract'."""
    if concept in CONCRETE_CONCEPTS:
        return "concrete"
    elif concept in ABSTRACT_CONCEPTS:
        return "abstract"
    return "unknown"
