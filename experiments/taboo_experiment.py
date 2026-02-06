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
from semiotic_channel.metrics import SemioticMetrics


def run_experiment(
    concepts: list[str] = None,
    temperatures: list[float] = None,
    n_descriptions_per_temp: int = 10,
    model_pairs: list[tuple[str, str]] = None,
    output_dir: str = "experiments/results",
    log_dir: str = "experiments/logs",
    verbose: bool = True,
    plotting: bool = True
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
        plotting: Plot the results
    
    Returns:
        Dictionary with experimental results
    """
    # Defaults
    if concepts is None:
        concepts = get_concepts("all")[:10]
    
    if temperatures is None:
        temperatures = [0.1, 0.4, 0.7, 1.0, 1.3, 1.6, 1.9]
        
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
    embeddings_model = OllamaEmbeddings() # For Clustering
    
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
                "S": [],        # Semiotic Breadth (Perplexity)
                "D": []         # Decipherability (Mutual Information)
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
                    
                # 2. Interpret (Full List for MI)
                guesses_lists = []
                
                for desc in descriptions:
                    guesses = receiver.interpret_top_k(desc, k=5)
                    guesses_lists.append(guesses)
                    
                    # Log Turn (Basic info)
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "emitter": emitter_model,
                        "receiver": receiver_model,
                        "concept": concept,
                        "temp": temp,
                        "description": desc,
                        "guesses": guesses
                    }
                    with open(log_file_path, 'a') as f:
                        f.write(json.dumps(log_entry) + "\n")
                
                # 3. Aggregate Metrics for Concept
                
                # Embed messages for clustering
                desc_embeddings = embeddings_model.embed_batch(descriptions)
                
                # Semiotic Breadth S (Perplexity)
                S = SemioticMetrics.breadth(desc_embeddings)
                
                # Decipherability D (Mutual Information)
                # We use the top-1 guess as the "interpretation" for the discrete side of MI
                top_1_guesses = [g[0] if g else "" for g in guesses_lists]
                D = SemioticMetrics.decipherability(desc_embeddings, top_1_guesses)
                
                if verbose:
                    print(f" S={S:.2f} (PPL), D={D:.2f} (MI)")
                
                # Store
                temp_metrics["S"].append(S)
                temp_metrics["D"].append(D)
                
                if concept not in pair_data["by_concept"]:
                    pair_data["by_concept"][concept] = {}
                pair_data["by_concept"][concept][temp] = {
                    "S": S, "D": D
                }
            
            # Aggregate for Temperature
            pair_data["by_temperature"][temp] = {
                "avg_S": float(np.mean(temp_metrics["S"])) if temp_metrics["S"] else 0,
                "avg_D": float(np.mean(temp_metrics["D"])) if temp_metrics["D"] else 0,
                "std_S": float(np.std(temp_metrics["S"])) if temp_metrics["S"] else 0,
                "std_D": float(np.std(temp_metrics["D"])) if temp_metrics["D"] else 0,
            }
            
        experiment_results["results"][pair_key] = pair_data

    # Save Results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"experiment_{timestamp_str}.json"
    
    with open(output_path, "w") as f:
        json.dump(experiment_results, f, indent=2, default=str)
    
    if verbose:
        print(f"\nResults saved to: {output_path}")
    
    if plotting:
        plot_results(experiment_results, output_path)
        
    return experiment_results


def plot_results(results: dict, output_path: str = "semiotic_profile_comparison.png"):
    """
    Generate visualization of Semiotic Breadth (S) and Decipherability (D) vs Temperature.
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, (pair_key, data) in enumerate(results["results"].items()):
        temps = sorted([float(t) for t in data["by_temperature"].keys()])
        S_values = [data["by_temperature"][str(t)]["avg_S"] for t in temps]
        D_values = [data["by_temperature"][str(t)]["avg_D"] for t in temps]
        
        color = colors[i % len(colors)]
        
        # Plot S line (Dashed)
        plt.plot(temps, S_values, marker='o', linestyle='--', color=color, alpha=0.7, label=f'{pair_key} (S - PPL)')
        
        # Plot D line (Solid)
        plt.plot(temps, D_values, marker='s', linestyle='-', color=color, linewidth=2, label=f'{pair_key} (D - MI)')

    plt.xlabel('Temperature (λ)')
    plt.ylabel('Score (Bits / PPL)')
    plt.title('Semiotic Channel Metrics by Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Taboo Semiotic Experiment")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model for Self-Play (default)")
    parser.add_argument("--emitter", help="Specific Ollama model for Emitter (overrides --model)")
    parser.add_argument("--receiver", help="Specific Ollama model for Receiver (overrides --model)")
    parser.add_argument("--n-concepts", type=int, default=5, help="Number of concepts to test")
    parser.add_argument("--n-descriptions", type=int, default=5, help="Descriptions per temperature")
    
    args = parser.parse_args()
    
    # Determine models
    emitter = args.emitter if args.emitter else args.model
    receiver = args.receiver if args.receiver else args.model
    
    pairs = [(emitter, receiver)]
    
    print(f"Configuration: Emitter={emitter}, Receiver={receiver}")
    
    run_experiment(
        concepts=get_concepts("all")[:args.n_concepts],
        n_descriptions_per_temp=args.n_descriptions,
        model_pairs=pairs,
        verbose=True
    )
