"""
Metrics for the Semiotic Channel Principle experiments.
Includes information-theoretic measures: Perplexity (S) and Mutual Information (D).
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


def calculate_perplexity(samples: list) -> float:
    """
    Calculates Perplexity of a distribution of samples.
    PPL(X) = 2^H(X)
    """
    h_x = calculate_entropy(samples)
    return 2.0 ** h_x


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


def calculate_clustered_mi(
    emitter_embeddings: list[list[float]], 
    receiver_interpretations: list[str],
    n_clusters: int = 5
) -> float:
    """
    Estimate Mutual Information between high-dimensional messages (embeddings)
    and discrete interpretations (strings).
    
    1. Clusters emitter_embeddings into n_clusters prototypes.
    2. Calculates I(M_cluster; Int) using the discrete interpretations.
    """
    if len(emitter_embeddings) < 2 or not receiver_interpretations:
        return 0.0
        
    from sklearn.cluster import KMeans
    
    # 1. Cluster embeddings to get discrete message states
    X = np.array(emitter_embeddings)
    n_clusters = min(n_clusters, len(emitter_embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    m_labels = kmeans.fit_predict(X).tolist()
    
    # 2. Compute MI between message clusters and discrete interpretations
    return calculate_mutual_information(m_labels, receiver_interpretations)


def calculate_embedding_perplexity(embeddings: list[list[float]], n_clusters: int = 5) -> float:
    """
    Calculate Perplexity of embeddings by clustering them.
    PPL = 2^Entropy
    
    Args:
        embeddings: List of embedding vectors
        n_clusters: Number of clusters for discretization
    
    Returns:
        Perplexity of the cluster distribution
    """
    if len(embeddings) < 2:
        return 0.0
    
    from sklearn.cluster import KMeans
    
    X = np.array(embeddings)
    n_clusters = min(n_clusters, len(embeddings))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    return calculate_perplexity(labels.tolist())


class SemioticMetrics:
    """Static methods for semiotic channel metrics."""
    
    @staticmethod
    def breadth(embeddings: list[list[float]]) -> float:
        """
        Semiotic Breadth S - measures diversity.
        Estimated via embedding perplexity: 2^H(M).
        """
        return calculate_embedding_perplexity(embeddings)

    @staticmethod
    def decipherability(
        emitter_embeddings: list[list[float]], 
        receiver_interpretations: list[str]
    ) -> float:
        """
        Decipherability D = I(M; Int).
        Estimated via clustered mutual information.
        """
        return calculate_clustered_mi(emitter_embeddings, receiver_interpretations)

    @staticmethod
    def capacity(decipherability_scores: list[float]) -> float:
        """
        Semiotic Capacity C = max(D(lambda))
        """
        return max(decipherability_scores) if decipherability_scores else 0.0

