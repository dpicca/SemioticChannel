"""
Metrics for the Semiotic Channel Principle experiments.
Includes information-theoretic measures and semantic similarity calculations.
"""

import numpy as np
from collections import Counter
from typing import Optional


def calculate_entropy(samples: list) -> float:
    """
    Calculates Shannon entropy of a distribution of samples.
    H(X) = -sum(p(x) * log2(p(x)))
    """
    if not samples:
        return 0.0
    
    counts = Counter(samples)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    
    return -sum(p * np.log2(p) for p in probs if p > 0)


def calculate_mutual_information(messages: list, interpretations: list) -> float:
    """
    Calculates Mutual Information between messages and interpretations.
    I(M; Int) = H(M) + H(Int) - H(M, Int)
    """
    if not messages or not interpretations or len(messages) != len(interpretations):
        return 0.0
    
    h_m = calculate_entropy(messages)
    h_int = calculate_entropy(interpretations)
    
    # Joint entropy H(M, Int)
    joint_samples = list(zip(messages, interpretations))
    h_m_int = calculate_entropy(joint_samples)
    
    return h_m + h_int - h_m_int


def calculate_residual_ambiguity(messages: list, interpretations: list) -> float:
    """
    Calculates Residual Ambiguity H(Int | M).
    H(Int | M) = H(M, Int) - H(M)
    """
    if not messages or not interpretations or len(messages) != len(interpretations):
        return 0.0
    
    h_m = calculate_entropy(messages)
    joint_samples = list(zip(messages, interpretations))
    h_m_int = calculate_entropy(joint_samples)
    
    return h_m_int - h_m


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec_a: First embedding vector
        vec_b: Second embedding vector
    
    Returns:
        Cosine similarity score (-1 to 1)
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def calculate_semantic_similarity(
    target_embedding: list[float], 
    interpreted_embedding: list[float]
) -> float:
    """
    Calculate Decipherability (D) as semantic similarity between
    the target concept and the interpreted word.
    
    D = cosine_similarity(embed(Target), embed(Interpreted))
    
    Args:
        target_embedding: Embedding of the original target concept
        interpreted_embedding: Embedding of the word guessed by receiver
    
    Returns:
        Similarity score (0 to 1, normalized from cosine)
    """
    sim = cosine_similarity(target_embedding, interpreted_embedding)
    # Normalize to 0-1 range (cosine can be -1 to 1)
    return (sim + 1) / 2


def calculate_reciprocal_rank(target: str, guesses: list[str]) -> float:
    """
    Calculate Reciprocal Rank (1/rank) of target in guesses.
    Rank is 1-based index (1st = 1.0, 2nd = 0.5).
    """
    target_clean = target.lower().strip()
    for i, guess in enumerate(guesses):
        if guess.lower().strip() == target_clean:
            return 1.0 / (i + 1)
    return 0.0


def calculate_self_bleu(descriptions: list[str], n_gram: int = 4) -> float:
    """
    Calculate Self-BLEU to measure diversity of generated descriptions.
    Lower Self-BLEU = Higher diversity = Higher Semiotic Breadth.
    
    We return (1 - Self-BLEU) so that higher values = more diversity.
    
    Args:
        descriptions: List of generated descriptions
        n_gram: N-gram size for BLEU calculation
    
    Returns:
        Diversity score (0 to 1)
    """
    if len(descriptions) < 2:
        return 0.0
    
    from collections import Counter
    
    def get_ngrams(text: str, n: int) -> Counter:
        words = text.lower().split()
        return Counter(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
    
    total_bleu = 0.0
    count = 0
    
    for i, hypothesis in enumerate(descriptions):
        references = [d for j, d in enumerate(descriptions) if j != i]
        
        hyp_ngrams = get_ngrams(hypothesis, n_gram)
        if not hyp_ngrams:
            continue
            
        # Calculate modified precision
        ref_ngrams = Counter()
        for ref in references:
            ref_ngrams.update(get_ngrams(ref, n_gram))
        
        clipped_count = sum(min(hyp_ngrams[ng], ref_ngrams.get(ng, 0)) for ng in hyp_ngrams)
        total_count = sum(hyp_ngrams.values())
        
        if total_count > 0:
            total_bleu += clipped_count / total_count
            count += 1
    
    self_bleu = total_bleu / count if count > 0 else 0.0
    
    # Return diversity (inverse of self-similarity)
    return 1.0 - self_bleu


def calculate_embedding_entropy(embeddings: list[list[float]], n_clusters: int = 5) -> float:
    """
    Calculate entropy of embeddings by clustering them.
    Higher entropy = more diverse embeddings = Higher Semiotic Breadth.
    
    Args:
        embeddings: List of embedding vectors
        n_clusters: Number of clusters for discretization
    
    Returns:
        Entropy of the cluster distribution
    """
    if len(embeddings) < 2:
        return 0.0
    
    from sklearn.cluster import KMeans
    
    X = np.array(embeddings)
    n_clusters = min(n_clusters, len(embeddings))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    return calculate_entropy(labels.tolist())


class SemioticMetrics:
    """Static methods for semiotic channel metrics."""
    
    @staticmethod
    def breadth(descriptions: list[str]) -> float:
        """
        Semiotic Breadth S(lambda) - measures lexical diversity.
        Uses Self-BLEU based diversity score.
        """
        return calculate_self_bleu(descriptions)

    @staticmethod
    def breadth_entropy(embeddings: list[list[float]]) -> float:
        """
        Alternative Semiotic Breadth using embedding entropy.
        """
        return calculate_embedding_entropy(embeddings)

    @staticmethod
    def decipherability(
        target_embedding: list[float],
        interpretation_embeddings: list[list[float]]
    ) -> float:
        """
        Decipherability D(lambda) - average semantic similarity
        between target and all interpretations.
        """
        if not interpretation_embeddings:
            return 0.0
        
        similarities = [
            calculate_semantic_similarity(target_embedding, interp_emb)
            for interp_emb in interpretation_embeddings
        ]
        return float(np.mean(similarities))

    @staticmethod
    def decipherability_rank(target: str, guesses_lists: list[list[str]]) -> float:
        """
        Rank-based Decipherability.
        Average Reciprocal Rank across multiple interpretation sets.
        """
        ranks = [calculate_reciprocal_rank(target, guesses) for guesses in guesses_lists]
        return float(np.mean(ranks)) if ranks else 0.0

    @staticmethod
    def capacity(decipherability_scores: list[float]) -> float:
        """
        Semiotic Capacity C = max(D(lambda))
        """
        return max(decipherability_scores) if decipherability_scores else 0.0
