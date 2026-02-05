"""
Taboo Semiotic Experiment: The Inverse Definition Game

Implements the experimental design to validate the Semiotic Channel Principle.
Measures how LLM temperature affects the trade-off between
Semiotic Breadth (S) and Decipherability (D).
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from semiotic_channel.concepts_dataset import get_concepts, get_concept_category
from semiotic_channel.llm_interface import (
    SemioticEmitter, 
    SemioticReceiver, 
    OllamaEmbeddings,
    check_ollama_available,
    list_available_models,
    DEFAULT_MODEL
)
from semiotic_channel.metrics import (
    SemioticMetrics, 
    calculate_reciprocal_rank, 
    calculate_efficiency
)


def run_experiment(
    concepts: list[str] = None,
    temperatures: list[float] = None,
    n_descriptions_per_temp: int = 10,
    model_pairs: list[tuple[str, str]] = None,
    output_dir: str = "experiments/results",
    log_dir: str = "experiments/logs",
    verbose: bool = True
):
    """
    Run the full Taboo Semiotic experiment with cross-model evaluation.
    
    Args:
        concepts: List of target concepts
        temperatures: List of temperature values to test
        n_descriptions_per_temp: Number of descriptions per concept per temperature
        model_pairs: List of (emitter, receiver) model tuples
        output_dir: Directory to save results
        log_dir: Directory to save turn-level logs
        verbose: Print progress
    
    Returns:
        Dictionary with experimental results
    """
    # Defaults
    if concepts is None:
        concepts = get_concepts("all")[:10]
    
    if temperatures is None:
        temperatures = [0.1, 0.4, 0.7, 1.0, 1.3]
        
    if model_pairs is None:
        # Default scenario: Self-play + Cross-play with stronger model
        model_pairs = [
            (DEFAULT_MODEL, DEFAULT_MODEL),
            # Add more pairs here if desired, e.g. (DEFAULT_MODEL, 'gemma:2b')
        ]

    # Check Ollama availability
    if not check_ollama_available():
        raise RuntimeError("Ollama server not running. Start it with 'ollama serve'")
    
    # Initialize components
    embeddings_model = OllamaEmbeddings() # For Semantic Diversity
    
    # Results storage
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_pairs": model_pairs,
            "temperatures": temperatures,
            "n_descriptions": n_descriptions_per_temp,
            "n_concepts": len(concepts)
        },
        "results": {}
    }
    
    # Setup Logging
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file_path = Path(log_dir) / f"experiment_turns_{timestamp_str}.jsonl"
    
    print(f"Logging turns to: {log_file_path}")

    # Main Loop over Model Pairs
    for emitter_model, receiver_model in model_pairs:
        pair_key = f"{emitter_model}+{receiver_model}"
        if verbose:
            print(f"\n{'#'*60}")
            print(f"Testing Pair: Emitter={emitter_model} -> Receiver={receiver_model}")
            print(f"{'#'*60}")
            
        emitter = SemioticEmitter(emitter_model)
        receiver = SemioticReceiver(receiver_model)
        
        pair_data = {
            "by_temperature": {},
            "by_concept": {}
        }
        
        for temp in temperatures:
            if verbose:
                print(f"\n  Temperature: {temp}")
            
            temp_metrics = {
                "breadths": [],        # Semantic Diversity
                "decipherabilities": [], # Reciprocal Rank
                "efficiencies": []     # Bits per success
            }
            
            for concept in concepts:
                if verbose:
                    print(f"    Concept: {concept}...", end="", flush=True)
                
                # 1. Generate Descriptions
                descriptions = []
                try:
                    descriptions = emitter.generate_batch(concept, temp, n_descriptions_per_temp)
                except Exception as e:
                    print(f" Error generating: {e}")
                    continue
                
                if not descriptions:
                    continue
                    
                # 2. Interpret (Ranked)
                guesses_lists = []
                turn_scores = []
                
                for desc in descriptions:
                    guesses = receiver.interpret_top_k(desc, k=5)
                    guesses_lists.append(guesses)
                    
                    # Turn Metrics
                    rank_score = calculate_reciprocal_rank(concept, guesses)
                    efficiency = calculate_efficiency(desc, rank_score)
                    turn_scores.append((rank_score, efficiency))
                    
                    # Log Turn
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "emitter": emitter_model,
                        "receiver": receiver_model,
                        "concept": concept,
                        "temp": temp,
                        "description": desc,
                        "guesses": guesses,
                        "rank": rank_score, # MRR for this turn
                        "efficiency": efficiency
                    }
                    with open(log_file_path, 'a') as f:
                        f.write(json.dumps(log_entry) + "\n")
                
                # 3. Aggregate Metrics for Concept
                
                # S: Semantic Diversity (needs embeddings)
                desc_embeddings = embeddings_model.embed_batch(descriptions)
                S = SemioticMetrics.diversity(desc_embeddings)
                
                # D: Decipherability (Rank-based)
                # Average of rank scores
                ranks = [s[0] for s in turn_scores]
                D =  float(np.mean(ranks)) if ranks else 0.0
                
                # E: Efficiency
                effs = [s[1] for s in turn_scores]
                E = float(np.mean(effs)) if effs else 0.0
                
                if verbose:
                    print(f" S={S:.2f}, D={D:.2f}, E={E:.2f}")
                
                # Store
                temp_metrics["breadths"].append(S)
                temp_metrics["decipherabilities"].append(D)
                temp_metrics["efficiencies"].append(E)
                
                if concept not in pair_data["by_concept"]:
                    pair_data["by_concept"][concept] = {}
                pair_data["by_concept"][concept][temp] = {"S": S, "D": D, "E": E}
            
            # Aggregate for Temperature
            pair_data["by_temperature"][temp] = {
                "avg_S": float(np.mean(temp_metrics["breadths"])) if temp_metrics["breadths"] else 0,
                "avg_D": float(np.mean(temp_metrics["decipherabilities"])) if temp_metrics["decipherabilities"] else 0,
                "avg_E": float(np.mean(temp_metrics["efficiencies"])) if temp_metrics["efficiencies"] else 0,
                "std_S": float(np.std(temp_metrics["breadths"])) if temp_metrics["breadths"] else 0,
                "std_D": float(np.std(temp_metrics["decipherabilities"])) if temp_metrics["decipherabilities"] else 0,
            }
            
        experiment_results["results"][pair_key] = pair_data

    # Save Results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"experiment_{timestamp_str}.json"
    
    with open(output_path, "w") as f:
        json.dump(experiment_results, f, indent=2, default=str)
    
    if verbose:
        print(f"\nResults saved to: {output_path}")
        
    return experiment_results


def plot_results(results: dict, output_path: str = "semiotic_profile_comparison.png"):
    """
    Generate visualization of the Semiotic Channel Profile for all pairs.
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    markers = ['o', 's', '^', 'D', 'x']
    
    for i, (pair_key, data) in enumerate(results["results"].items()):
        temps = sorted([float(t) for t in data["by_temperature"].keys()])
        S_values = [data["by_temperature"][t]["avg_S"] for t in temps]
        D_values = [data["by_temperature"][t]["avg_D"] for t in temps]
        
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Plot 1: D vs Temperature
        ax1.plot(temps, D_values, marker=marker, linestyle='-', color=color, label=f'{pair_key} (D)')
        # Could also plot S, but might get crowded. Let's stick to D comparison or S vs D.
        
        # Plot 2: Semiotic Profile (S vs D)
        ax2.plot(S_values, D_values, marker=marker, linestyle='-', color=color, label=pair_key, linewidth=2)
        
        # Annotate points
        for j, temp in enumerate(temps):
            if j == 0 or j == len(temps)-1: # Only start/end to avoid clutter
                ax2.annotate(f'{temp}', (S_values[j], D_values[j]), 
                            textcoords="offset points", xytext=(5, 5), fontsize=8, color=color)

    # Styling
    ax1.set_xlabel('Temperature (λ)')
    ax1.set_ylabel('Decipherability (D)')
    ax1.set_title('Decipherability by Temperature')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Semantic Diversity (S)')
    ax2.set_ylabel('Decipherability (D)')
    ax2.set_title('Semiotic Channel Profile (Breadth vs Depth)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Taboo Semiotic Experiment")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model for Self-Play")
    parser.add_argument("--n-concepts", type=int, default=5, help="Number of concepts to test")
    parser.add_argument("--n-descriptions", type=int, default=5, help="Descriptions per temperature")
    
    args = parser.parse_args()
    
    # Simple self-play default
    pairs = [(args.model, args.model)]
    
    run_experiment(
        concepts=get_concepts("all")[:args.n_concepts],
        n_descriptions_per_temp=args.n_descriptions,
        model_pairs=pairs,
        verbose=True
    )
