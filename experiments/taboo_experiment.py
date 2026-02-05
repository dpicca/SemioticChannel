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
    DEFAULT_MODEL,
    RECEIVER_MODEL
)
from semiotic_channel.metrics import SemioticMetrics, calculate_semantic_similarity


def run_experiment(
    concepts: list[str] = None,
    temperatures: list[float] = None,
    n_descriptions_per_temp: int = 10,
    model_emitter: str = DEFAULT_MODEL,
    model_receiver: str = RECEIVER_MODEL,
    output_dir: str = "experiments/results",
    verbose: bool = True
):
    """
    Run the full Taboo Semiotic experiment.
    
    Args:
        concepts: List of target concepts (default: all 100)
        temperatures: List of temperature values to test
        n_descriptions_per_temp: Number of descriptions per concept per temperature
        model_emitter: str = DEFAULT_MODEL,
    model_receiver: Ollama model to use
        output_dir: Directory to save results
        verbose: Print progress
    
    Returns:
        Dictionary with experimental results
    """
    # Defaults
    if concepts is None:
        concepts = get_concepts("all")[:10]  # Use 10 for testing
    
    if temperatures is None:
        temperatures = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    
    # Check Ollama availability
    if not check_ollama_available():
        raise RuntimeError("Ollama server not running. Start it with 'ollama serve'")
    
    available_models = list_available_models()
    if verbose:
        print(f"Available models: {available_models}")
    
    # Initialize components
    emitter = SemioticEmitter(model_emitter)
    receiver = SemioticReceiver(model_receiver)
    embeddings = OllamaEmbeddings()
    
    # Results storage
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model_emitter": model_emitter,
            "model_receiver": model_receiver,
            "temperatures": temperatures,
            "n_descriptions": n_descriptions_per_temp,
            "n_concepts": len(concepts)
        },
        "by_temperature": {},
        "by_concept": {}
    }
    
    # Pre-compute target embeddings
    if verbose:
        print("\nPre-computing target embeddings...")
    target_embeddings = {}
    for concept in concepts:
        target_embeddings[concept] = embeddings.embed(concept)
    
    # Main experimental loop
    for temp in temperatures:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Temperature: {temp}")
            print(f"{'='*60}")
        
        temp_results = {
            "breadths": [],
            "decipherabilities": [],
            "concepts_data": []
        }
        
        for concept in concepts:
            if verbose:
                print(f"\n  Concept: {concept} ({get_concept_category(concept)})")
            
            # Phase A: Generate descriptions
            if verbose:
                print(f"    Generating {n_descriptions_per_temp} descriptions...")
            
            descriptions = []
            for i in range(n_descriptions_per_temp):
                try:
                    desc = emitter.generate_description(concept, temperature=temp)
                    descriptions.append(desc)
                    if verbose and i == 0:
                        print(f"    Sample: \"{desc[:80]}...\"" if len(desc) > 80 else f"    Sample: \"{desc}\"")
                except Exception as e:
                    print(f"    Error generating description: {e}")
                    continue
            
            if not descriptions:
                continue
            
            # Phase B: Interpret descriptions
            if verbose:
                print(f"    Interpreting descriptions...")
            
            interpretations = []
            for desc in descriptions:
                try:
                    interp = receiver.interpret(desc)
                    interpretations.append(interp)
                except Exception as e:
                    print(f"    Error interpreting: {e}")
                    interpretations.append("")
            
            if verbose:
                interp_sample = interpretations[:3]
                print(f"    Interpretations sample: {interp_sample}")
            
            # Phase C: Calculate metrics
            
            # Semiotic Breadth (S) - lexical diversity of descriptions
            S = SemioticMetrics.breadth(descriptions)
            
            # Decipherability (D) - semantic similarity between target and interpretations
            target_emb = target_embeddings[concept]
            interp_embeddings = []
            for interp in interpretations:
                if interp:
                    try:
                        interp_emb = embeddings.embed(interp)
                        interp_embeddings.append(interp_emb)
                    except Exception:
                        pass
            
            D = SemioticMetrics.decipherability(target_emb, interp_embeddings)
            
            if verbose:
                print(f"    S (Breadth): {S:.4f}")
                print(f"    D (Decipherability): {D:.4f}")
            
            # Store results
            temp_results["breadths"].append(S)
            temp_results["decipherabilities"].append(D)
            temp_results["concepts_data"].append({
                "concept": concept,
                "category": get_concept_category(concept),
                "descriptions": descriptions,
                "interpretations": interpretations,
                "S": S,
                "D": D
            })
            
            # Also store by concept
            if concept not in results["by_concept"]:
                results["by_concept"][concept] = {}
            results["by_concept"][concept][temp] = {"S": S, "D": D}
        
        # Aggregate metrics for this temperature
        avg_S = np.mean(temp_results["breadths"]) if temp_results["breadths"] else 0
        avg_D = np.mean(temp_results["decipherabilities"]) if temp_results["decipherabilities"] else 0
        
        results["by_temperature"][temp] = {
            "avg_S": float(avg_S),
            "avg_D": float(avg_D),
            "std_S": float(np.std(temp_results["breadths"])) if temp_results["breadths"] else 0,
            "std_D": float(np.std(temp_results["decipherabilities"])) if temp_results["decipherabilities"] else 0,
            "concepts_data": temp_results["concepts_data"]
        }
        
        if verbose:
            print(f"\n  Avg S: {avg_S:.4f}, Avg D: {avg_D:.4f}")
    
    # Calculate Semiotic Capacity
    all_D = [results["by_temperature"][t]["avg_D"] for t in temperatures]
    capacity = SemioticMetrics.capacity(all_D)
    capacity_temp = temperatures[np.argmax(all_D)]
    
    results["capacity"] = {
        "C": float(capacity),
        "optimal_temperature": float(capacity_temp)
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SEMIOTIC CAPACITY: {capacity:.4f}")
        print(f"Optimal Temperature: {capacity_temp}")
        print(f"{'='*60}")
    
    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"experiment_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    if verbose:
        print(f"\nResults saved to: {output_path}")
    
    return results


def plot_results(results: dict, output_path: str = "semiotic_profile.png"):
    """
    Generate visualization of the Semiotic Channel Profile.
    """
    import matplotlib.pyplot as plt
    
    temps = sorted([float(t) for t in results["by_temperature"].keys()])
    S_values = [results["by_temperature"][t]["avg_S"] for t in temps]
    D_values = [results["by_temperature"][t]["avg_D"] for t in temps]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: S and D vs Temperature
    ax1.plot(temps, S_values, 'b-o', label='S (Breadth)', linewidth=2)
    ax1.plot(temps, D_values, 'r-s', label='D (Decipherability)', linewidth=2)
    
    # Mark capacity point
    opt_temp = results["capacity"]["optimal_temperature"]
    opt_D = results["capacity"]["C"]
    ax1.axvline(x=opt_temp, color='green', linestyle='--', alpha=0.7, label=f'Optimal λ={opt_temp}')
    ax1.scatter([opt_temp], [opt_D], color='green', s=100, zorder=5)
    
    ax1.set_xlabel('Temperature (λ)', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Semiotic Breadth vs Decipherability', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add zone labels
    ax1.axvspan(0, 0.4, alpha=0.1, color='blue', label='Rigidity Zone')
    ax1.axvspan(0.5, 1.0, alpha=0.1, color='green', label='Optimal Zone')
    ax1.axvspan(1.1, max(temps), alpha=0.1, color='red', label='Hallucination Zone')
    
    # Plot 2: S vs D (Semiotic Channel Profile)
    ax2.plot(S_values, D_values, 'purple', linewidth=2)
    ax2.scatter(S_values, D_values, c=temps, cmap='viridis', s=80, zorder=5)
    
    for i, temp in enumerate(temps):
        ax2.annotate(f'λ={temp}', (S_values[i], D_values[i]), 
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    ax2.set_xlabel('Semiotic Breadth (S)', fontsize=12)
    ax2.set_ylabel('Decipherability (D)', fontsize=12)
    ax2.set_title('Semiotic Channel Profile', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Mark capacity
    opt_idx = temps.index(opt_temp)
    ax2.scatter([S_values[opt_idx]], [D_values[opt_idx]], 
               color='red', s=150, marker='*', zorder=10, label=f'Capacity C={opt_D:.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Taboo Semiotic Experiment")
    parser.add_argument("--model-emitter", default=DEFAULT_MODEL, help="Ollama model to use as emitter")
    parser.add_argument("--model-receiver", default=RECEIVER_MODEL, help="Ollama model to use as receiver")
    parser.add_argument("--n-concepts", type=int, default=5, help="Number of concepts to test")
    parser.add_argument("--n-descriptions", type=int, default=5, help="Descriptions per temperature")
    parser.add_argument("--category", choices=["all", "concrete", "abstract"], default="all")
    
    args = parser.parse_args()
    
    concepts = get_concepts(args.category)[:args.n_concepts]
    
    print(f"Running experiment with {len(concepts)} concepts using {args.model_emitter}")
    print(f"Concepts: {concepts}")
    
    results = run_experiment(
        concepts=concepts,
        n_descriptions_per_temp=args.n_descriptions,
        model_emitter=args.model_emitter,
        model_receiver=args.model_receiver,
        verbose=True
    )
    
    plot_results(results)
